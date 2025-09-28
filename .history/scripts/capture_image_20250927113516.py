while True:
    ret, frame = cap.read()
    print(f"Frame read status: {ret}")  # Add this line

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    cv2.imshow("Capture", frame)

    # Save images as before...
