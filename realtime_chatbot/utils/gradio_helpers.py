def get_audio_html(reset_elem_id, audio_elem_id):
    audio_html = f'''
        <script type="text/javascript">
            var lastSrc = "";
            function checkAudio() {{
                try {{
                    var ce = window.parent.document.getElementsByTagName("gradio-app")[0];
                    var ce_dom = ce.shadowRoot ?? window.parent.document;
                    var reset = ce_dom.getElementById("{reset_elem_id}").querySelector("input");
                    if (!(reset && reset.checked)) {{
                        var audio = ce_dom.getElementById("{audio_elem_id}").querySelector("audio");
                        if (audio && audio.src !== lastSrc) {{
                            lastSrc = audio.src;
                            var audio_clone = audio.cloneNode();
                            audio_clone.onended = function() {{
                                if (this.nextSibling && this.nextSibling.nodeName === "AUDIO") {{
                                    this.nextSibling.play();
                                }}
                                this.parentNode.removeChild(this);
                            }};
                            ce.parentNode.appendChild(audio_clone);
                            if (!audio_clone.previousSibling || audio_clone.previousSibling.nodeName !== "AUDIO") {{
                                audio_clone.play();
                            }}
                        }}
                    }} else {{
                        var existing_audios = ce.parentNode.querySelectorAll("audio");
                        for (var i = 0; i < existing_audios.length; i++) {{
                            ce.parentNode.removeChild(existing_audios[i]);
                        }}
                    }}
                }} catch (ex) {{
                    console.error(ex);
                }}
                setTimeout(checkAudio, 100);
            }}
            checkAudio();
        </script>
    '''
    audio_html = f'''<iframe style="width: 100%; height: 0px" frameborder="0" srcdoc='{audio_html}'></iframe>'''
    return audio_html