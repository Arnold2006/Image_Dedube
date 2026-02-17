import gradio as gr
import os
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import shutil

# Try to import sentence_transformers for CLIP embeddings
try:
    from sentence_transformers import SentenceTransformer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("sentence_transformers not available. Will use basic image comparison.")

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SimilarImageFinder:
    def __init__(self):
        self.model = None
        self.image_embeddings = {}
        self.image_paths = []
        self.similar_groups = []
        
        if CLIP_AVAILABLE:
            try:
                # Load CLIP model for image embeddings
                self.model = SentenceTransformer('clip-ViT-B-32')
                print("Loaded CLIP model successfully")
            except Exception as e:
                print(f"Could not load CLIP model: {e}")
                self.model = None
    
    def load_images_from_folder(self, folder_path: str) -> List[str]:
        """Load all image files from the specified folder."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        image_paths = []
        
        folder = Path(folder_path)
        if not folder.exists():
            return []
        
        for file in folder.rglob('*'):
            if file.suffix.lower() in supported_formats:
                image_paths.append(str(file))
        
        return sorted(image_paths)
    
    def compute_embeddings(self, image_paths: List[str], progress=gr.Progress()) -> Dict[str, np.ndarray]:
        """Compute embeddings for all images."""
        embeddings = {}
        
        if self.model is None:
            # Fallback: use basic image comparison (histogram)
            for i, img_path in enumerate(progress.tqdm(image_paths, desc="Computing image features")):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((64, 64))  # Small size for speed
                    embeddings[img_path] = np.array(img.histogram()).astype(np.float32)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        else:
            # Use CLIP embeddings
            for i, img_path in enumerate(progress.tqdm(image_paths, desc="Computing image embeddings")):
                try:
                    img = Image.open(img_path).convert('RGB')
                    embedding = self.model.encode(img)
                    embeddings[img_path] = embedding
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return embeddings
    
    def find_similar_groups(self, embeddings: Dict[str, np.ndarray], threshold: float = 0.85) -> List[List[str]]:
        """Group similar images together."""
        paths = list(embeddings.keys())
        emb_array = np.array([embeddings[p] for p in paths])
        
        # Normalize embeddings
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        emb_array = emb_array / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(emb_array, emb_array.T)
        
        # Find groups
        visited = set()
        groups = []
        
        for i, path1 in enumerate(paths):
            if i in visited:
                continue
            
            group = [path1]
            visited.add(i)
            
            for j, path2 in enumerate(paths):
                if i != j and j not in visited and similarity_matrix[i, j] >= threshold:
                    group.append(path2)
                    visited.add(j)
            
            if len(group) > 1:  # Only include groups with more than 1 image
                groups.append(group)
        
        return groups
    
    def analyze_folder(self, folder_path: str, similarity_threshold: float, progress=gr.Progress()):
        """Main function to analyze a folder and find similar images."""
        if not folder_path or not os.path.exists(folder_path):
            return "❌ Invalid folder path", None, None
        
        # Load images
        progress(0, desc="Loading images...")
        self.image_paths = self.load_images_from_folder(folder_path)
        
        if len(self.image_paths) == 0:
            return "❌ No images found in folder", None, None
        
        # Compute embeddings
        progress(0.2, desc="Analyzing images...")
        self.image_embeddings = self.compute_embeddings(self.image_paths, progress)
        
        # Find similar groups
        progress(0.8, desc="Finding similar images...")
        self.similar_groups = self.find_similar_groups(self.image_embeddings, similarity_threshold)
        
        if len(self.similar_groups) == 0:
            return f"✓ Analyzed {len(self.image_paths)} images. No similar groups found.", None, None
        
        summary = f"✓ Analyzed {len(self.image_paths)} images. Found {len(self.similar_groups)} groups of similar images."
        
        # Return the first group for display
        return summary, 0, gr.update(maximum=len(self.similar_groups)-1, value=0)


def create_interface():
    finder = SimilarImageFinder()
    
    with gr.Blocks(title="Similar Image Finder", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🖼️ Similar Image Finder & Cleanup Tool")
        gr.Markdown("Find groups of similar images and select which ones to delete.")
        
        with gr.Row():
            with gr.Column(scale=2):
                folder_input = gr.Textbox(
                    label="📁 Image Folder Path",
                    placeholder="/path/to/your/images",
                    lines=1
                )
            with gr.Column(scale=1):
                similarity_slider = gr.Slider(
                    minimum=0.5,
                    maximum=0.99,
                    value=0.85,
                    step=0.05,
                    label="🎯 Similarity Threshold",
                    info="Higher = more strict matching"
                )
        
        analyze_btn = gr.Button("🔍 Analyze Folder", variant="primary", size="lg")
        status_text = gr.Textbox(label="Status", interactive=False, lines=2)
        
        gr.Markdown("---")
        gr.Markdown("## 📋 Review Similar Image Groups")
        
        with gr.Row():
            group_slider = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="Group Number",
                interactive=True
            )
        
        group_info = gr.Markdown("")
        
        with gr.Row():
            gallery = gr.Gallery(
                label="Similar Images (click to select for deletion)",
                show_label=True,
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                selected_index=None
            )
        
        with gr.Row():
            selected_indices = gr.State([])
            selection_text = gr.Textbox(
                label="Selected for Deletion",
                interactive=False,
                lines=2
            )
        
        with gr.Row():
            delete_btn = gr.Button("🗑️ Delete Selected Images", variant="stop")
            keep_best_btn = gr.Button("✨ Keep Largest, Delete Others", variant="secondary")
        
        delete_status = gr.Textbox(label="Deletion Status", interactive=False, lines=2)
        
        # State variables
        current_group = gr.State(0)
        
        def display_group(group_idx):
            if group_idx >= len(finder.similar_groups) or group_idx < 0:
                return "", [], "No images to display", []
            
            group = finder.similar_groups[int(group_idx)]
            info = f"**Group {int(group_idx) + 1} of {len(finder.similar_groups)}** — {len(group)} similar images"
            
            return info, group, "", []
        
        def update_selection(evt: gr.SelectData, current_selected):
            selected = current_selected.copy() if current_selected else []
            idx = evt.index
            
            if idx in selected:
                selected.remove(idx)
            else:
                selected.append(idx)
            
            if not selected:
                return selected, "No images selected"
            
            group_idx = group_slider.value
            if group_idx >= len(finder.similar_groups):
                return selected, "Invalid group"
            
            group = finder.similar_groups[int(group_idx)]
            selected_files = [Path(group[i]).name for i in sorted(selected) if i < len(group)]
            
            return selected, f"Selected {len(selected_files)} images:\n" + "\n".join(selected_files)
        
        def delete_selected(group_idx, selected):
            if not selected or group_idx >= len(finder.similar_groups):
                return "❌ No images selected or invalid group", []
            
            group = finder.similar_groups[int(group_idx)]
            deleted = []
            errors = []
            
            for idx in selected:
                if idx < len(group):
                    try:
                        file_path = group[idx]
                        os.remove(file_path)
                        deleted.append(Path(file_path).name)
                    except Exception as e:
                        errors.append(f"{Path(group[idx]).name}: {str(e)}")
            
            result = f"✓ Deleted {len(deleted)} images"
            if errors:
                result += f"\n❌ Errors: {len(errors)}\n" + "\n".join(errors[:5])
            
            # Remove deleted files from the group
            finder.similar_groups[int(group_idx)] = [p for i, p in enumerate(group) if i not in selected]
            
            return result, []
        
        def keep_largest(group_idx):
            if group_idx >= len(finder.similar_groups):
                return "❌ Invalid group"
            
            group = finder.similar_groups[int(group_idx)]
            if len(group) <= 1:
                return "❌ Group has only one image"
            
            # Find the largest image by file size
            largest_idx = 0
            largest_size = 0
            
            for i, path in enumerate(group):
                try:
                    size = os.path.getsize(path)
                    if size > largest_size:
                        largest_size = size
                        largest_idx = i
                except:
                    pass
            
            # Delete all except the largest
            deleted = []
            errors = []
            
            for i, path in enumerate(group):
                if i != largest_idx:
                    try:
                        os.remove(path)
                        deleted.append(Path(path).name)
                    except Exception as e:
                        errors.append(f"{Path(path).name}: {str(e)}")
            
            result = f"✓ Kept largest image, deleted {len(deleted)} others"
            if errors:
                result += f"\n❌ Errors: {len(errors)}"
            
            # Update the group to only contain the kept image
            finder.similar_groups[int(group_idx)] = [group[largest_idx]]
            
            return result
        
        # Event handlers
        analyze_btn.click(
            fn=finder.analyze_folder,
            inputs=[folder_input, similarity_slider],
            outputs=[status_text, current_group, group_slider]
        ).then(
            fn=display_group,
            inputs=[group_slider],
            outputs=[group_info, gallery, selection_text, selected_indices]
        )
        
        group_slider.change(
            fn=display_group,
            inputs=[group_slider],
            outputs=[group_info, gallery, selection_text, selected_indices]
        )
        
        gallery.select(
            fn=update_selection,
            inputs=[selected_indices],
            outputs=[selected_indices, selection_text]
        )
        
        delete_btn.click(
            fn=delete_selected,
            inputs=[group_slider, selected_indices],
            outputs=[delete_status, selected_indices]
        ).then(
            fn=display_group,
            inputs=[group_slider],
            outputs=[group_info, gallery, selection_text, selected_indices]
        )
        
        keep_best_btn.click(
            fn=keep_largest,
            inputs=[group_slider],
            outputs=[delete_status]
        ).then(
            fn=display_group,
            inputs=[group_slider],
            outputs=[group_info, gallery, selection_text, selected_indices]
        )
    
    return app


if __name__ == "__main__":
    if not CLIP_AVAILABLE:
        print("\n⚠️  Warning: sentence-transformers not installed.")
        print("For better results, install it with:")
        print("pip install sentence-transformers --break-system-packages")
        print("\nUsing basic image comparison as fallback.\n")
    
    app = create_interface()
    app.launch(share=False)
