import gradio as gr
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageSequenceClip, AudioFileClip
from gtts import gTTS
import torch

# Load the AI image generator
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.enable_attention_slicing()
pipe.scheduler.config.num_inference_steps = 20


# Generate a smooth-transition video
def make_video(prompt):
    print("🎨 Generating images...")
    images = []
    for i in range(1, 11):  # 10 frames
        frame_prompt = f"{prompt}, cinematic lighting, frame {i}"
        image = pipe(frame_prompt).images[0]
        filename = f"frame_{i}.png"
        image.save(filename)
        images.append(filename)

    # 🔊 Text-to-speech audio from prompt
    print("🎤 Generating audio...")
    tts = gTTS(f"This is a short video of: {prompt}", lang='en')
    tts.save("audio.mp3")

    # 🎬 Make the video from images
    print("🎞 Creating video...")
    clip = ImageSequenceClip(images, fps=1)
    audio = AudioFileClip("audio.mp3").set_duration(clip.duration)
    video = clip.set_audio(audio)
    video.write_videofile("output_video.mp4", codec="libx264", audio_codec="aac")

    print("✅ Done!")
    return "output_video.mp4"

# Gradio UI
gr.Interface(
    fn=make_video,
    inputs="text",
    outputs=gr.Video(label="Generated Video", format="mp4"),
    title="Smooth AI Video Maker 🎬",
    description="Type a prompt. It will create a 10-second video with smooth scene changes and audio.",
    allow_flagging="never"
).launch(share=True)
