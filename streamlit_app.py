# import os
# import streamlit as st
# import pandas as pd
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch
# torch.cuda.empty_cache()

# # Specify the model name
# model_name = "Salesforce/blip-image-captioning-large"

# # Load the Blip processor and model
# processor = BlipProcessor.from_pretrained(model_name)
# model = BlipForConditionalGeneration.from_pretrained(model_name).to("cuda")

# # Function to process images and generate prompts
# def process_images(images, output_csv_path):
    # # Initialize an empty DataFrame to store the results
    # columns = ['Image File', 'Conditional Prompt', 'Unconditional Prompt']
    # results_df = pd.DataFrame(columns=columns)

    # # Iterate through each image in the folder
    # for filename, image_data in images.items():
        # # Load the image
        # raw_image = Image.open(image_data).convert('RGB')

        # # Conditional image captioning
        # text = "a photography of"
        # inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

        # out = model.generate(**inputs)
        # conditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # # Unconditional image captioning
        # inputs = processor(raw_image, return_tensors="pt").to("cuda")

        # out = model.generate(**inputs)
        # unconditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # # Append the results to the DataFrame
        # results_df = results_df.append({'Image File': filename, 'Conditional Prompt': conditional_prompt, 'Unconditional Prompt': unconditional_prompt}, ignore_index=True)

    # # Save the results to the specified CSV file
    # results_df.to_csv(output_csv_path, index=False)

    # # Display the generated prompts
    # st.table(results_df)

# # Streamlit app layout
# st.title("Image Captioning Streamlit App")

# # Allow user to upload a zip file containing images
# uploaded_file = st.file_uploader("Upload a zip file containing images", type=["zip"])

# # Allow user to choose output CSV file path
# output_csv_path = st.text_input("Enter Output CSV File Path", value="output.csv")

# # Button to process images
# if st.button("Process Images") and uploaded_file is not None:
    # # Extract images from the zip file
    # with st.spinner("Extracting images..."):
        # images = {}
        # with zipfile.ZipFile(uploaded_file) as archive:
            # for file_info in archive.infolist():
                # with archive.open(file_info) as file:
                    # images[file_info.filename] = file.read()

    # # Process images
    # with st.spinner("Processing images..."):
        # process_images(images, output_csv_path)
        
# import os
# import streamlit as st
# import pandas as pd
# import torch 
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Specify the model name
# model_name = "Salesforce/blip-image-captioning-large"

# # Load the Blip processor and model
# processor = BlipProcessor.from_pretrained(model_name)
# model = BlipForConditionalGeneration.from_pretrained(model_name).to("cuda")

# # Function to process images and generate prompts
# def process_images(images, output_csv_path, batch_size=2):  # Adjust the batch size as needed
    # # Initialize an empty DataFrame to store the results
    # columns = ['Image File', 'Conditional Prompt', 'Unconditional Prompt']
    # results_df = pd.DataFrame(columns=columns)

    # # Iterate through each image in the folder
    # for filename, image_data in images.items():
        # # Load the image
        # raw_image = Image.open(image_data).convert('RGB')

        # # Conditional image captioning
        # text = "a photography of"
        # inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

        # out = model.generate(**inputs, batch_size=batch_size)  # Use the specified batch size
        # conditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # # Unconditional image captioning
        # inputs = processor(raw_image, return_tensors="pt").to("cuda")

        # out = model.generate(**inputs, batch_size=batch_size)  # Use the specified batch size
        # unconditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # # Append the results to the DataFrame
        # results_df = results_df.append({'Image File': filename, 'Conditional Prompt': conditional_prompt, 'Unconditional Prompt': unconditional_prompt}, ignore_index=True)
        
        # # Clear GPU memory
        # torch.cuda.empty_cache()

    # # Save the results to the specified CSV file
    # results_df.to_csv(output_csv_path, index=False)

    # # Display the generated prompts
    # st.table(results_df)

# # Streamlit app layout
# st.title("Image Captioning Streamlit App")

# # Allow user to upload multiple image files
# image_files = st.sidebar.file_uploader("Select Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# # Allow user to choose output CSV file path
# output_csv_path = st.text_input("Enter Output CSV File Path", value="output.csv")

# # Button to process images
# if st.button("Process Images") and image_files:
    # # Extract images from the uploaded files
    # with st.spinner("Processing images..."):
        # images = {}
        # for uploaded_file in image_files:
            # # Ensure the uploaded file is an image
            # if uploaded_file.type.startswith('image'):
                # image_data = uploaded_file.read()
                # images[uploaded_file.name] = image_data

    # # Process images
    # process_images(images, output_csv_path)

# import os
# import streamlit as st
# import pandas as pd
# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Specify the model name
# model_name = "Salesforce/blip-image-captioning-large"

# # Load the Blip processor and model
# processor = BlipProcessor.from_pretrained(model_name)
# model = BlipForConditionalGeneration.from_pretrained(model_name).to("cuda")

# # Function to process images and generate prompts
# def process_images(images, output_csv_path, initial_batch_size=1, max_batch_size=16):
    # # Initialize an empty DataFrame to store the results
    # columns = ['Image File', 'Conditional Prompt', 'Unconditional Prompt']
    # results_df = pd.DataFrame(columns=columns)

    # # Iterate through different batch sizes
    # for batch_size in range(initial_batch_size, max_batch_size + 1):
        # try:
            # # Iterate through each image in the folder
            # for filename, image_data in images.items():
                # # Load the image
                # raw_image = Image.open(image_data).convert('RGB')

                # # Conditional image captioning
                # text = "a photography of"
                # inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

                # out = model.generate(**inputs)
                # conditional_prompt = processor.decode(out[0], skip_special_tokens=True)

                # # Unconditional image captioning
                # inputs = processor(raw_image, return_tensors="pt").to("cuda")

                # out = model.generate(**inputs)
                # unconditional_prompt = processor.decode(out[0], skip_special_tokens=True)

                # # Append the results to the DataFrame
                # results_df = results_df.append({'Image File': filename, 'Conditional Prompt': conditional_prompt, 'Unconditional Prompt': unconditional_prompt}, ignore_index=True)

            # # If successful, break the loop
            # break
        # except torch.cuda.CudaError:
            # # If CUDA error occurs, reduce the batch size and try again
            # torch.cuda.empty_cache()
            # continue
        # finally:
            # # Clear GPU memory after processing each image
            # torch.cuda.empty_cache()

    # # Save the results to the specified CSV file
    # results_df.to_csv(output_csv_path, index=False)

    # # Display the generated prompts
    # st.table(results_df)

# # Streamlit app layout
# st.title("Image Captioning Streamlit App")

# # Allow user to upload multiple image files
# image_files = st.sidebar.file_uploader("Select Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# # Allow user to choose output CSV file path
# output_csv_path = st.text_input("Enter Output CSV File Path", value="output.csv")

# # Slider to adjust the initial batch size
# initial_batch_size = st.sidebar.slider("Initial Batch Size", min_value=1, max_value=16, value=1)

# # Button to process images
# if st.button("Process Images") and image_files:
    # # Extract images from the uploaded files
    # with st.spinner("Processing images..."):
        # images = {}
        # for uploaded_file in image_files:
            # # Ensure the uploaded file is an image
            # if uploaded_file.type.startswith('image'):
                # image_data = uploaded_file.read()
                # images[uploaded_file.name] = image_data

    # # Process images with adjustable batch size
    # process_images(images, output_csv_path, initial_batch_size=initial_batch_size)
    
# import os
# import streamlit as st
# import pandas as pd
# import torch 
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Specify the model name
# model_name = "Salesforce/blip-image-captioning-large"

# # Load the Blip processor and model
# processor = BlipProcessor.from_pretrained(model_name)
# model = BlipForConditionalGeneration.from_pretrained(model_name).to("cuda")

# # Function to process images and generate prompts
# def process_images(images, output_csv_path):
    # # Initialize an empty DataFrame to store the results
    # columns = ['Image File', 'Conditional Prompt', 'Unconditional Prompt']
    # results_df = pd.DataFrame(columns=columns)

    # # Iterate through each image in the folder
    # for filename, image_data in images.items():
        # # Load the image
        # raw_image = Image.open(image_data).convert('RGB')

        # # Conditional image captioning
        # text = "a photography of"
        # inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

        # with torch.no_grad():
            # out = model.generate(**inputs)
        
        # conditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # # Unconditional image captioning
        # inputs = processor(raw_image, return_tensors="pt").to("cuda")

        # with torch.no_grad():
            # out = model.generate(**inputs)
        
        # unconditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # # Append the results to the DataFrame
        # results_df = results_df.append({'Image File': filename, 'Conditional Prompt': conditional_prompt, 'Unconditional Prompt': unconditional_prompt}, ignore_index=True)
        
        # # Clear GPU memory
        # torch.cuda.empty_cache()

    # # Save the results to the specified CSV file
    # results_df.to_csv(output_csv_path, index=False)

    # # Display the generated prompts
    # st.table(results_df)

# # Streamlit app layout
# st.title("Image Captioning Streamlit App")

# # Allow user to upload multiple image files
# image_files = st.sidebar.file_uploader("Select Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# # Allow user to choose output CSV file path
# output_csv_path = st.text_input("Enter Output CSV File Path", value="output.csv")

# # Button to process images
# if st.button("Process Images") and image_files:
    # # Extract images from the uploaded files
    # with st.spinner("Processing images..."):
        # images = {}
        # for uploaded_file in image_files:
            # # Ensure the uploaded file is an image
            # if uploaded_file.type.startswith('image'):
                # image_data = uploaded_file.read()
                # images[uploaded_file.name] = image_data

    # # Process images
    # process_images(images, output_csv_path)

import os
import streamlit as st
import pandas as pd
import torch 
import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Image Captioning Streamlit App')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# Specify the model name
model_name = "Salesforce/blip-image-captioning-large"

# Load the Blip processor and model
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(args.device)

# Function to process images and generate prompts
def process_images(images, output_csv_path):
    # Initialize an empty DataFrame to store the results
    columns = ['Image File', 'Conditional Prompt', 'Unconditional Prompt']
    results_df = pd.DataFrame(columns=columns)

    # Iterate through each image in the folder
    for filename, image_data in images.items():
        # Load the image
        raw_image = Image.open(image_data).convert('RGB')

        # Conditional image captioning
        text = "a photography of"
        inputs = processor(raw_image, text, return_tensors="pt").to(args.device)

        with torch.no_grad():
            out = model.generate(**inputs)
        
        conditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to(args.device)

        with torch.no_grad():
            out = model.generate(**inputs)
        
        unconditional_prompt = processor.decode(out[0], skip_special_tokens=True)

        # Append the results to the DataFrame
        results_df = results_df.append({'Image File': filename, 'Conditional Prompt': conditional_prompt, 'Unconditional Prompt': unconditional_prompt}, ignore_index=True)
        
        # Clear GPU memory
        torch.cuda.empty_cache()

    # Save the results to the specified CSV file
    results_df.to_csv(output_csv_path, index=False)

    # Display the generated prompts
    st.table(results_df)

# Streamlit app layout
st.title("Image Captioning Streamlit App")

# Allow user to upload multiple image files
image_files = st.sidebar.file_uploader("Select Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Allow user to choose output CSV file path
output_csv_path = st.text_input("Enter Output CSV File Path", value="output.csv")

# Button to process images
if st.button("Process Images") and image_files:
    # Extract images from the uploaded files
    with st.spinner("Processing images..."):
        images = {}
        for uploaded_file in image_files:
            # Ensure the uploaded file is an image
            if uploaded_file.type.startswith('image'):
                image_data = uploaded_file.read()
                images[uploaded_file.name] = image_data

    # Process images
    process_images(images, output_csv_path)

