from flask import Flask, render_template, request, send_from_directory
import os
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageSearcher:
    def __init__(self, embedding_path="image_embeddings.pickle", image_folder="coco_images_resized"):
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        self.image_folder = image_folder
        self.df = pd.read_pickle(embedding_path)
        self.filenames = self.df["file_name"].values
        self.embeddings = np.stack(self.df["embedding"].values)

        # Load CLIP model
        self.model_name = "ViT-B-32"
        self.pretrained = "openai"
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Precompute PCA embeddings
        self.pca = PCA(n_components=50)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings)

    def encode_text(self, text):
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(device)
            text_emb = self.model.encode_text(tokens)
            return text_emb.cpu().numpy().squeeze()

    def encode_image(self, image_file):
        image = Image.open(image_file).convert("RGB")
        image_tensor = self.preprocess_val(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = self.model.encode_image(image_tensor)
        return img_emb.cpu().numpy().squeeze()

    def combine_embeddings(self, text_emb, image_emb, lam=0.5):
        combined = lam * text_emb + (1 - lam) * image_emb
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined

    def search(self, query_emb, top_k=5, use_pca=False):
        if use_pca:
            query_emb = self.pca.transform(query_emb.reshape(1, -1))[0]
            query_emb = query_emb / np.linalg.norm(query_emb)  
            embeddings = self.embeddings_pca
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) 
        else:
            query_emb = query_emb / np.linalg.norm(query_emb)
            embeddings = self.embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        sim_scores = embeddings @ query_emb
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        return [(self.filenames[i], sim_scores[i]) for i in top_indices]


searcher = ImageSearcher(image_folder="coco_images_resized")

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error_message = None

    if request.method == "POST":
        query_type = request.form.get("query_type", "text")
        text_query = request.form.get("text_query", "").strip()
        use_pca = request.form.get("use_pca") == "on"
        image_query_file = request.files.get("image_query")
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5)) 

        try:
            if query_type == "text" and text_query:
                query_emb = searcher.encode_text(text_query)
            elif query_type == "image" and image_query_file:
                query_emb = searcher.encode_image(image_query_file)
            elif query_type == "hybrid" and text_query and image_query_file:
                text_emb = searcher.encode_text(text_query)
                image_emb = searcher.encode_image(image_query_file)
                query_emb = searcher.combine_embeddings(text_emb, image_emb, lam=hybrid_weight)
            else:
                raise ValueError("Invalid input for the selected query type.")

            results = searcher.search(query_emb, top_k=5, use_pca=use_pca)
        except Exception as e:
            error_message = str(e)

    return render_template("index.html", results=results, error_message=error_message)


@app.route("/coco_images_resized/<path:filename>")
def serve_image(filename):
    image_folder = "coco_images_resized"
    if not os.path.exists(os.path.join(image_folder, filename)):
        return f"Image {filename} not found in {image_folder}", 404
    return send_from_directory("coco_images_resized", filename)


if __name__ == "__main__":
    app.run(debug=True)
