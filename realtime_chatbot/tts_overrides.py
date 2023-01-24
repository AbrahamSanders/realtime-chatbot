from typing import Optional, Tuple
from fairseq.data.data_utils import lengths_to_padding_mask
import torch

def get_phonemize_override(cls, g2p):

    def phonemize(
        text: str,
        lang: Optional[str],
        phonemizer: Optional[str] = None,
        preserve_punct: bool = False,
        to_simplified_zh: bool = False,
    ):
        if preserve_punct:
            return " ".join("|" if p == " " else p for p in g2p(text))
        else:
            res = [{",": "sp", ";": "sp"}.get(p, p) for p in g2p(text)]
            return " ".join(p for p in res if p.isalnum())

    return phonemize

def get_get_prediction_override(cls):

    def get_prediction(task, model, generator, sample, **kwargs) -> Tuple[torch.Tensor, int]:
        prediction = generator.generate(model, sample, **kwargs)
        return prediction[0]["waveform"], task.sr

    return get_prediction

def get_generate_override(self):

    @torch.no_grad()
    def generate(model, sample, has_targ=False, **kwargs):
        model.eval()

        bsz, max_src_len = sample["net_input"]["src_tokens"].size()
        n_frames_per_step = model.encoder.n_frames_per_step
        out_dim = model.encoder.out_dim
        raw_dim = out_dim // n_frames_per_step

        feat, feat_post, out_lens, log_dur_out, _, _ = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            incremental_state=None,
            target_lengths=sample["target_lengths"],
            speaker=sample["speaker"],
            **kwargs
        )
        if feat_post is not None:
            feat = feat_post

        feat = feat.view(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)

        dur_out = torch.clamp(torch.round(torch.exp(log_dur_out) - 1).long(), min=0)

        def get_dur_plot_data(d):
            r = []
            for i, dd in enumerate(d):
                r += [i + 1] * dd.item()
            return r

        out_lens = out_lens * n_frames_per_step
        finalized = [
            {
                "feature": feat[b, :l] if l > 0 else feat.new_zeros([1, raw_dim]),
                "waveform": self.get_waveform(
                    feat[b, :l] if l > 0 else feat.new_zeros([1, raw_dim])
                ),
                "attn": feat.new_tensor(get_dur_plot_data(dur_out[b])),
            }
            for b, l in zip(range(bsz), out_lens)
        ]

        if has_targ:
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            tgt_lens = sample["target_lengths"] * n_frames_per_step
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])
        return finalized

    return generate

def get_encoder_forward_override(self):

    def forward(
        src_tokens,
        src_lengths=None,
        speaker=None,
        durations=None,
        pitches=None,
        energies=None,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
        **kwargs,
    ):
        x = self.embed_tokens(src_tokens)

        enc_padding_mask = src_tokens.eq(self.padding_idx)
        x += self.pos_emb_alpha * self.embed_positions(enc_padding_mask)
        x = self.dropout_module(x)

        for layer in self.encoder_fft_layers:
            x = layer(x, enc_padding_mask)

        if self.embed_speaker is not None:
            bsz, seq_len, _ = x.size()
            emb = self.embed_speaker(speaker).expand(bsz, seq_len, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        x, out_lens, log_dur_out, pitch_out, energy_out = self.var_adaptor(
            x, enc_padding_mask, durations, pitches, energies, d_factor, p_factor, e_factor
        )

        dec_padding_mask = lengths_to_padding_mask(out_lens)
        x += self.dec_pos_emb_alpha * self.embed_positions(dec_padding_mask)
        for layer in self.decoder_fft_layers:
            x = layer(x, dec_padding_mask)

        x = self.out_proj(x)
        x_post = None
        if self.postnet is not None:
            x_post = x + self.postnet(x)
        return x, x_post, out_lens, log_dur_out, pitch_out, energy_out

    return forward