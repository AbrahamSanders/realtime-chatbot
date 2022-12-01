import uuid

def get_audio_html(audio_elem_id):
    audio_html = f'''
        <script type="text/javascript">
            var ce = window.parent.document.getElementsByTagName("gradio-app")[0];
            var audio = ce.shadowRoot.getElementById("{audio_elem_id}").querySelector("audio");
            var audio_clone = audio.cloneNode();
            var ce_parent = ce.parentNode;
            ce_parent.appendChild(audio_clone);

            if (audio_clone.previousSibling.nodeName !== "AUDIO") {{
                audio_clone.play();
            }} else if (audio_clone.previousSibling.paused && audio_clone.previousSibling.previousSibling.nodeName !== "AUDIO") {{
                ce_parent.removeChild(audio_clone.previousSibling);
                audio_clone.play();
            }} else {{
                audio_clone.previousSibling.onended = function() {{
                    ce_parent.removeChild(audio_clone.previousSibling);
                    audio_clone.play();
                }};
            }}
        </script>
    '''
    unique_name = str(uuid.uuid4())[:8]
    audio_html = f"""
        <iframe style="width: 100%; height: 0px" name="{unique_name}" frameborder="0" 
            srcdoc='{audio_html}'></iframe>"""

    return audio_html