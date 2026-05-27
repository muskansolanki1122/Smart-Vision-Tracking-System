from ultralytics import YOLO
import cv2
import streamlit as st

# TITLE
st.title("YOLO Detection + Tracking Dashboard")

# Load model
model = YOLO("yolo26n.pt")

# Upload video
video_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if video_file is not None:

    # Save uploaded file
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    # Open video file
    cap = cv2.VideoCapture("temp.mp4")

    # Video placeholder
    stframe = st.empty()

    # Counting setup
    counted_ids = set()

    line_y = 300

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Run tracking
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        # Draw counting line
        cv2.line(
            frame,
            (0, line_y),
            (frame.shape[1], line_y),
            (0, 0, 255),
            2
        )

        # Process detections
        if results[0].boxes is not None:

            for box in results[0].boxes:

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Class name
                cls = int(box.cls[0])

                name = model.names[cls]

                # Tracking ID
                track_id = (
                    int(box.id[0])
                    if box.id is not None
                    else -1
                )

                # Center point
                cx = int((x1 + x2) / 2)

                cy = int((y1 + y2) / 2)

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                cv2.putText(
                    frame,
                    f"{name} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # Draw center point
                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    -1
                )

                # COUNT LOGIC
                if line_y - 10 < cy < line_y + 10:
                    counted_ids.add(track_id)

        # Show count
        cv2.putText(
            frame,
            f"Count: {len(counted_ids)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # Convert BGR to RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Display frame
        stframe.image(
            frame,
            channels="RGB"
        )

    cap.release()

    st.success("Processing Completed")