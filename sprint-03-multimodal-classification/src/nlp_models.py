""" Evaluate Medical Tests Classification in LLMS """
import os

import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch


class GPT:
    """
    A class to interact with the OpenAI GPT API for generating text embeddings from a given dataset.

    Args:
        path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
        embedding_model (str, optional): The embedding model to use for generating text embeddings.
                                         Default is 'text-embedding-3-small'.
    """
    def __init__(self, path='data/file.csv', embedding_model='text-embedding-3-small'):
        import openai
        from dotenv import load_dotenv, find_dotenv
        _ = load_dotenv(find_dotenv())
        openai.api_key = os.environ.get('OPENAI_API_KEY')

        self.path = path
        self.embedding_model = embedding_model

    def get_embedding(self, text):
        """
        Generates and returns the embedding vector for the given text using the OpenAI API.

        Args:
            text (str): The input text to generate the embedding for.

        Returns:
            list: A list containing the embedding vector for the input text.
        """
        from openai import OpenAI
        client = OpenAI()

        text = text.replace("\n", " ")

        embeddings_np = client.embeddings.create(input=[text], model=self.embedding_model).data[0].embedding
        return embeddings_np

    def get_embedding_df(self, column, directory, file):
        """
        Reads a CSV file, computes the embeddings for a specified text column, and saves the results in a new CSV file.

        Args:
            column (str): The name of the column in the CSV file that contains the text data.
            directory (str): The directory where the output CSV file will be saved.
            file (str): The name of the output CSV file.
        """
        df = pd.read_csv(self.path)
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x))

        os.makedirs(directory, exist_ok=True)
        df.to_csv(os.path.join(directory, file), index=False)


class HuggingFaceEmbeddings:
    """
    A class to handle text embedding generation using a Hugging Face pre-trained transformer model.

    Args:
        model_name (str, optional): The name of the Hugging Face pre-trained model to use for generating embeddings.
                                    Default is 'sentence-transformers/all-MiniLM-L6-v2'.
        path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
        save_path (str, optional): The directory path where the embeddings will be saved. Default is 'Models'.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). If None, it will automatically detect
                                a GPU if available; otherwise, it defaults to CPU.
    """

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', path='data/file.csv', save_path=None, device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.path = path
        self.save_path = save_path or 'Models'

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")
        print(f"Model: {model_name}")

    def get_embedding(self, text):
        """
        Generates embeddings for a given text using the Hugging Face model.

        Args:
            text (str): The input text for which embeddings will be generated.

        Returns:
            np.ndarray: A numpy array containing the embedding vector for the input text.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embeddings

    def get_embedding_df(self, column, directory, file):
        df = pd.read_csv(self.path)
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x).tolist())

        os.makedirs(directory, exist_ok=True)
        df.to_csv(os.path.join(directory, file), index=False)
