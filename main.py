import cv2
import dlib
import numpy as np
import face_recognition
from tkinter import Tk, Button, Label
from PIL import Image, ImageTk
import logging

# Setup logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def select_image(option):
    global selected_image_path
    selected_image_path = image_options[option]
    root.destroy()


def face_swap(user_image_path, template_image_path):
    logging.info("Loading images.")
    # Load the images
    user_image = face_recognition.load_image_file(user_image_path)
    template_image = face_recognition.load_image_file(template_image_path)

    # Convert images from RGB to BGR for OpenCV
    user_image = user_image[:, :, ::-1]
    template_image = template_image[:, :, ::-1]

    logging.info("Finding faces in the user image.")
    # Find faces and face encodings in the user image
    user_face_locations = face_recognition.face_locations(user_image)
    if not user_face_locations:
        logging.warning("No faces found in the user image.")
        return None  # No faces found

    user_face_encoding = face_recognition.face_encodings(
        user_image, user_face_locations
    )[0]

    logging.info("Finding faces in the template image.")
    # Find faces in the template image
    template_face_locations = face_recognition.face_locations(template_image)
    template_face_encodings = face_recognition.face_encodings(
        template_image, template_face_locations
    )

    if not template_face_encodings:
        logging.warning("No faces found in the template image.")
        return template_image  # Return unchanged template if no faces

    logging.info("Calculating face distances to find the best match.")
    # Calculate the face with the closest match to the user's face
    face_distances = face_recognition.face_distance(
        template_face_encodings, user_face_encoding
    )
    best_match_index = np.argmin(face_distances)

    # Coordinates for the face in the template
    top, right, bottom, left = template_face_locations[best_match_index]

    logging.info("Resizing user's face and creating mask.")
    # Scale and mask user's face to fit on template
    user_face_height, user_face_width = bottom - top, right - left
    user_face_image = cv2.resize(user_image, (user_face_width, user_face_height))

    # Create a mask for the user's face
    mask = np.zeros((user_face_height, user_face_width, 3), dtype=np.uint8)
    mask.fill(255)  # White mask

    # Place user's face on the template
    combined = template_image.copy()
    combined[top:bottom, left:right] = cv2.seamlessClone(
        user_face_image,
        combined[top:bottom, left:right],
        mask,
        (user_face_width // 2, user_face_height // 2),
        cv2.NORMAL_CLONE,
    )

    logging.info("Face swap complete.")
    return combined


# GUI setup
root = Tk()
root.title("Select an Image Template")

# image_paths = [
#     "images/image1.jpg",
#     "images/image2.jpg",
#     "images/image3.jpg",
#     "images/image4.jpg",
# ]
# image_options = {str(i): path for i, path in enumerate(image_paths, start=1)}
# buttons = []

# for idx, path in image_options.items():
#     img = Image.open(path)
#     img.thumbnail((100, 100))
#     img = ImageTk.PhotoImage(img)
#     button = Button(root, image=img, command=lambda option=idx: select_image(option))
#     button.image = img
#     button.pack(side="left")
#     buttons.append(button)

# root.mainloop()

# # Face swap the selected image with the user's face
# user_image_path = "images/myFace.jpg"
# swapped_image = face_swap(user_image_path, selected_image_path)
# cv2.imshow("Face Swapped Image", swapped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Example usage
try:
    swapped_image = face_swap("images/myFace.jpg", "images/image1.jpg")
    if swapped_image is not None:
        cv2.imshow("Face Swapped Image", swapped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logging.error("Face swap failed.")
except Exception as e:
    logging.error("An error occurred: {}".format(e))
