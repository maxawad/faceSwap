from openai import OpenAI

client = OpenAI()


def generate_single_image(prompt):
    response = client.images.generate(
        model="dall-e-3", prompt=prompt, n=1, size="1024x1024"
    )
    return response


def generate_images(prompt, n_images):
    # generate n_images amount of images and return
    images = []
    for i in range(n_images):
        image = generate_single_image(prompt)
        images.append(image)
    return images


# Define your prompt


# Number of images to generate
num_images = 4

# Call the function to generate images


def clean_up_urls(images):
    cleaned_urls = []
    for image in images:
        clean_url = image.data[0].url
        cleaned_urls.append(clean_url)
    return cleaned_urls


# Print the response or handle it as needed


def __main__(
    prompt="A man standing by the beach during the day. The beach is scenic with clear blue skies, gentle waves, and soft sand. The man is casually dressed in a light t-shirt and shorts, with sunglasses on his head, looking relaxed and enjoying the ocean breeze. He is of medium build and has short brown hair.",
):
    prompt_description = prompt
    images_response = generate_images(prompt_description, num_images)
    print(clean_up_urls(images_response))

    return clean_up_urls(images_response)
