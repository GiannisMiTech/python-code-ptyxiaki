from ultralytics import YOLO
from moviepy.editor import VideoFileClip, CompositeAudioClip, AudioFileClip

# Initialize model
model = YOLO("last (train3).pt")

# Process video and get results
results = model(source='video2.mp4' , conf=0.55, show=True, save=True)

# Load the original video to get the frame rate
video = VideoFileClip('video2_box.avi')
frame_rate = video.fps
# Extract events with times and corresponding sound files
events = []
for frame_id, r in enumerate(results):
    if any(cls == 0 for cls in r.boxes.cls):  # Check if class 0 (cane class) is detected cane sound effect 
        events.append((frame_id / frame_rate, 'Ding Sound Effect.mp3'))
    if any(cls == 1 for cls in r.boxes.cls):  # Check if class 1 (guide dog class) is detected  dog sound effect
        events.append((frame_id / frame_rate, 'Tick Sound.mp3'))


# Function to add sound effects at specific times
def add_sound_effects(video, events):
    audio_clips = []

    for event_time, sound_file in events:
        sound = AudioFileClip(sound_file).set_start(event_time)
        audio_clips.append(sound)

    new_audio = CompositeAudioClip(audio_clips)
    return video.set_audio(new_audio)

# Add the sound effects to the video
video_with_sounds = add_sound_effects(video, events)

# Save the new video with sound effects
output_path = 'video2_with_sounds.mp4'
video_with_sounds.write_videofile(output_path, codec='libx264', audio_codec='aac')





