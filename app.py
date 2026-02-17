import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
import numpy as np
from typing import List, Dict
import threading

# Try to import sentence_transformers for CLIP embeddings
try:
    from sentence_transformers import SentenceTransformer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using basic comparison.")


class SimilarImageFinder:
    def __init__(self):
        self.model = None
        self.image_embeddings = {}
        self.image_paths = []
        self.similar_groups = []
        
        if CLIP_AVAILABLE:
            try:
                self.model = SentenceTransformer('clip-ViT-B-32')
                print("✓ Loaded CLIP model successfully")
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
    
    def compute_embeddings(self, image_paths: List[str], progress_callback=None) -> Dict[str, np.ndarray]:
        """Compute embeddings for all images."""
        embeddings = {}
        total = len(image_paths)
        
        if self.model is None:
            # Fallback: use basic image comparison (histogram)
            for i, img_path in enumerate(image_paths):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((64, 64))
                    embeddings[img_path] = np.array(img.histogram()).astype(np.float32)
                    if progress_callback:
                        progress_callback(i + 1, total)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        else:
            # Use CLIP embeddings
            for i, img_path in enumerate(image_paths):
                try:
                    img = Image.open(img_path).convert('RGB')
                    embedding = self.model.encode(img)
                    embeddings[img_path] = embedding
                    if progress_callback:
                        progress_callback(i + 1, total)
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
            
            if len(group) > 1:
                groups.append(group)
        
        return groups


class ImageDedupeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Similar Image Finder & Cleanup Tool")
        self.root.geometry("1200x800")
        
        self.finder = SimilarImageFinder()
        self.current_group_idx = 0
        self.selected_indices = set()
        self.photo_images = []  # Keep references to prevent garbage collection
        
        self.create_widgets()
        
        if not CLIP_AVAILABLE:
            messagebox.showwarning(
                "Limited Features",
                "sentence-transformers not installed.\n\n"
                "For better results, install with:\n"
                "pip install sentence-transformers\n\n"
                "Using basic image comparison as fallback."
            )
    
    def create_widgets(self):
        # Top frame - folder selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N))
        
        ttk.Label(top_frame, text="Image Folder:").grid(row=0, column=0, sticky=tk.W)
        
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(top_frame, textvariable=self.folder_var, width=60)
        folder_entry.grid(row=0, column=1, padx=5)
        
        browse_btn = ttk.Button(top_frame, text="Browse...", command=self.browse_folder)
        browse_btn.grid(row=0, column=2)
        
        ttk.Label(top_frame, text="Similarity Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=0.85)
        threshold_scale = ttk.Scale(top_frame, from_=0.5, to=0.99, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.threshold_label = ttk.Label(top_frame, text="0.85")
        self.threshold_label.grid(row=1, column=2)
        self.threshold_var.trace('w', self.update_threshold_label)
        
        # Add helpful note about threshold
        threshold_note = ttk.Label(top_frame, 
                                   text="💡 0.50-0.70 = loose matching (finds somewhat similar)\n"
                                        "   0.85 = recommended (finds very similar images)\n"
                                        "   0.90-0.99 = strict (only nearly identical images)",
                                   foreground="gray", font=("Arial", 8))
        threshold_note.grid(row=2, column=1, sticky=tk.W, pady=(0, 5))
        
        analyze_btn = ttk.Button(top_frame, text="🔍 Analyze Folder", command=self.analyze_folder)
        analyze_btn.grid(row=3, column=1, pady=10)
        
        self.status_var = tk.StringVar(value="Ready. Select a folder to begin.")
        status_label = ttk.Label(top_frame, textvariable=self.status_var, foreground="blue")
        status_label.grid(row=4, column=0, columnspan=3, sticky=tk.W)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(top_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Middle frame - group navigation
        nav_frame = ttk.Frame(self.root, padding="10")
        nav_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(nav_frame, text="Similar Image Groups:").grid(row=0, column=0, sticky=tk.W)
        
        btn_frame = ttk.Frame(nav_frame)
        btn_frame.grid(row=0, column=1, padx=20)
        
        self.prev_btn = ttk.Button(btn_frame, text="← Previous", command=self.prev_group, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=0, padx=5)
        
        self.group_label_var = tk.StringVar(value="No groups yet")
        ttk.Label(btn_frame, textvariable=self.group_label_var).grid(row=0, column=1, padx=10)
        
        self.next_btn = ttk.Button(btn_frame, text="Next →", command=self.next_group, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=2, padx=5)
        
        # Image display frame
        display_frame = ttk.Frame(self.root, padding="10")
        display_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create canvas with scrollbar for images
        canvas_frame = ttk.Frame(display_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW)
        
        self.image_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Bind mouse wheel scrolling for Windows, Mac, and Linux
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows and Mac
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)    # Linux scroll down
        
        # Bottom frame - actions
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
        self.selection_var = tk.StringVar(value="No images selected")
        ttk.Label(bottom_frame, textvariable=self.selection_var).grid(row=0, column=0, columnspan=3, pady=5)
        
        delete_selected_btn = ttk.Button(bottom_frame, text="🗑️ Delete Selected Images", 
                                        command=self.delete_selected, state=tk.DISABLED)
        delete_selected_btn.grid(row=1, column=0, padx=5)
        self.delete_selected_btn = delete_selected_btn
        
        keep_largest_btn = ttk.Button(bottom_frame, text="✨ Keep Largest, Delete Others", 
                                     command=self.keep_largest, state=tk.DISABLED)
        keep_largest_btn.grid(row=1, column=1, padx=5)
        self.keep_largest_btn = keep_largest_btn
        
        clear_selection_btn = ttk.Button(bottom_frame, text="Clear Selection", 
                                        command=self.clear_selection, state=tk.DISABLED)
        clear_selection_btn.grid(row=1, column=2, padx=5)
        self.clear_selection_btn = clear_selection_btn
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
    
    def update_threshold_label(self, *args):
        self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling cross-platform."""
        if event.num == 4 or event.delta > 0:  # Scroll up (Linux: Button-4, Win/Mac: positive delta)
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down (Linux: Button-5, Win/Mac: negative delta)
            self.canvas.yview_scroll(1, "units")
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folder_var.set(folder)
    
    def analyze_folder(self):
        folder = self.folder_var.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder")
            return
        
        # Disable buttons during analysis
        self.status_var.set("Loading images...")
        self.progress_var.set(0)
        self.root.update()
        
        # Run analysis in a thread to keep UI responsive
        thread = threading.Thread(target=self._analyze_thread, args=(folder,))
        thread.daemon = True
        thread.start()
    
    def _analyze_thread(self, folder):
        try:
            # Load images
            self.finder.image_paths = self.finder.load_images_from_folder(folder)
            
            if len(self.finder.image_paths) == 0:
                self.root.after(0, lambda: self.status_var.set("❌ No images found in folder"))
                self.root.after(0, lambda: self.progress_var.set(0))
                return
            
            self.root.after(0, lambda: self.status_var.set(f"Analyzing {len(self.finder.image_paths)} images..."))
            
            # Compute embeddings with progress updates
            def update_progress(current, total):
                progress = (current / total) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
            
            self.finder.image_embeddings = self.finder.compute_embeddings(
                self.finder.image_paths, 
                progress_callback=update_progress
            )
            
            # Find similar groups
            self.root.after(0, lambda: self.status_var.set("Finding similar images..."))
            threshold = self.threshold_var.get()
            self.finder.similar_groups = self.finder.find_similar_groups(
                self.finder.image_embeddings, 
                threshold
            )
            
            # Update UI
            if len(self.finder.similar_groups) == 0:
                self.root.after(0, lambda: self.status_var.set(
                    f"✓ Analyzed {len(self.finder.image_paths)} images. No similar groups found."
                ))
                self.root.after(0, lambda: self.progress_var.set(0))
            else:
                self.current_group_idx = 0
                self.root.after(0, lambda: self.status_var.set(
                    f"✓ Found {len(self.finder.similar_groups)} groups of similar images"
                ))
                self.root.after(0, self.display_current_group)
                self.root.after(0, lambda: self.prev_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.next_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("❌ Analysis failed"))
        finally:
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def display_current_group(self):
        # Clear previous images
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.photo_images.clear()
        self.selected_indices.clear()
        
        if not self.finder.similar_groups or self.current_group_idx >= len(self.finder.similar_groups):
            return
        
        group = self.finder.similar_groups[self.current_group_idx]
        self.group_label_var.set(f"Group {self.current_group_idx + 1} of {len(self.finder.similar_groups)}")
        self.selection_var.set("No images selected (click images to select)")
        
        # Display images in a grid
        cols = 4
        for idx, img_path in enumerate(group):
            try:
                # Load and resize image
                img = Image.open(img_path)
                img.thumbnail((250, 250), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.photo_images.append(photo)
                
                # Create frame for each image with white background by default
                frame = tk.Frame(self.image_frame, relief=tk.SOLID, borderwidth=3, bg="white", cursor="hand2")
                frame.grid(row=idx // cols, column=idx % cols, padx=5, pady=5)
                
                # Container for image and checkmark overlay
                img_container = tk.Frame(frame, bg="white", cursor="hand2")
                img_container.pack(padx=5, pady=5)
                
                # Image label
                img_label = tk.Label(img_container, image=photo, cursor="hand2", bg="white")
                img_label.pack()
                
                # Checkmark label (initially hidden)
                checkmark_label = tk.Label(img_container, text="✓ SELECTED", 
                                          font=("Arial", 12, "bold"), 
                                          fg="white", bg="green",
                                          relief=tk.RAISED, borderwidth=2,
                                          cursor="hand2")
                checkmark_label.place(x=0, y=0)  # Top-left corner
                checkmark_label.place_forget()  # Hide initially
                
                # File info with white background
                file_name = Path(img_path).name
                file_size = os.path.getsize(img_path) / 1024  # KB
                info_text = f"{file_name}\n{file_size:.1f} KB"
                info_label = tk.Label(frame, text=info_text, justify=tk.CENTER, 
                                     font=("Arial", 8), bg="white", cursor="hand2")
                info_label.pack(padx=5, pady=(0, 5))
                
                # Bind click event to all interactive elements
                def on_click(i):
                    return lambda e: self.toggle_selection(i)
                
                def on_enter(widget):
                    return lambda e: widget.config(bg="#e6f3ff") if idx not in self.selected_indices else None
                
                def on_leave(widget):
                    return lambda e: widget.config(bg="white") if idx not in self.selected_indices else None
                
                img_label.bind("<Button-1>", on_click(idx))
                frame.bind("<Button-1>", on_click(idx))
                checkmark_label.bind("<Button-1>", on_click(idx))
                img_container.bind("<Button-1>", on_click(idx))
                info_label.bind("<Button-1>", on_click(idx))
                
                # Hover effects
                for widget in [frame, img_label, img_container, info_label]:
                    widget.bind("<Enter>", on_enter(widget))
                    widget.bind("<Leave>", on_leave(widget))
                
                # Store references for selection highlighting
                frame.image_index = idx
                frame.checkmark_label = checkmark_label
                frame.img_container = img_container
                frame.info_label = info_label
                
            except Exception as e:
                print(f"Error displaying {img_path}: {e}")
        
        # Enable action buttons
        self.delete_selected_btn.config(state=tk.NORMAL)
        self.keep_largest_btn.config(state=tk.NORMAL)
        self.clear_selection_btn.config(state=tk.NORMAL)
        
        # Update canvas scroll region
        self.image_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def toggle_selection(self, idx):
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.add(idx)
        
        # Update visual feedback with prominent colors
        for widget in self.image_frame.winfo_children():
            if hasattr(widget, 'image_index'):
                if widget.image_index in self.selected_indices:
                    # Selected: red border, green checkmark, light red background
                    widget.config(relief=tk.SOLID, borderwidth=5, bg="#ffcccc")
                    widget.checkmark_label.place(x=5, y=5)  # Show checkmark
                    widget.img_container.config(bg="#ffcccc")
                    widget.info_label.config(bg="#ffcccc", fg="red", font=("Arial", 8, "bold"))
                else:
                    # Not selected: normal appearance
                    widget.config(relief=tk.SOLID, borderwidth=3, bg="white")
                    widget.checkmark_label.place_forget()  # Hide checkmark
                    widget.img_container.config(bg="white")
                    widget.info_label.config(bg="white", fg="black", font=("Arial", 8))
        
        # Update selection text
        if not self.selected_indices:
            self.selection_var.set("No images selected (click images to select)")
        else:
            group = self.finder.similar_groups[self.current_group_idx]
            selected_files = [Path(group[i]).name for i in sorted(self.selected_indices)]
            self.selection_var.set(f"Selected {len(selected_files)} images: " + ", ".join(selected_files[:3]) + 
                                  ("..." if len(selected_files) > 3 else ""))
    
    def clear_selection(self):
        self.selected_indices.clear()
        self.display_current_group()
    
    def delete_selected(self):
        if not self.selected_indices:
            messagebox.showwarning("No Selection", "Please select images to delete")
            return
        
        group = self.finder.similar_groups[self.current_group_idx]
        selected_files = [Path(group[i]).name for i in sorted(self.selected_indices)]
        
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {len(selected_files)} images?\n\n" + "\n".join(selected_files[:5]) +
            ("\n..." if len(selected_files) > 5 else "") +
            "\n\nThis cannot be undone!"
        )
        
        if not confirm:
            return
        
        deleted = []
        errors = []
        
        for idx in self.selected_indices:
            try:
                file_path = group[idx]
                os.remove(file_path)
                deleted.append(Path(file_path).name)
            except Exception as e:
                errors.append(f"{Path(group[idx]).name}: {str(e)}")
        
        # Update group
        self.finder.similar_groups[self.current_group_idx] = [
            p for i, p in enumerate(group) if i not in self.selected_indices
        ]
        
        # Show result
        result = f"✓ Deleted {len(deleted)} images"
        if errors:
            result += f"\n❌ Errors: {len(errors)}"
            messagebox.showwarning("Deletion Complete", result)
        else:
            messagebox.showinfo("Success", result)
        
        # Refresh display or move to next group
        if len(self.finder.similar_groups[self.current_group_idx]) <= 1:
            # Remove this group if only 1 or 0 images left
            self.finder.similar_groups.pop(self.current_group_idx)
            if self.current_group_idx >= len(self.finder.similar_groups):
                self.current_group_idx = max(0, len(self.finder.similar_groups) - 1)
        
        if self.finder.similar_groups:
            self.display_current_group()
        else:
            self.status_var.set("✓ All groups processed!")
            self.clear_display()
    
    def keep_largest(self):
        group = self.finder.similar_groups[self.current_group_idx]
        
        if len(group) <= 1:
            messagebox.showinfo("Info", "Group has only one image")
            return
        
        confirm = messagebox.askyesno(
            "Keep Largest?",
            f"Keep the largest image and delete {len(group) - 1} others?\n\nThis cannot be undone!"
        )
        
        if not confirm:
            return
        
        # Find largest
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
        
        # Delete all except largest
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
            messagebox.showwarning("Deletion Complete", result)
        else:
            messagebox.showinfo("Success", result)
        
        # Remove group
        self.finder.similar_groups.pop(self.current_group_idx)
        if self.current_group_idx >= len(self.finder.similar_groups):
            self.current_group_idx = max(0, len(self.finder.similar_groups) - 1)
        
        if self.finder.similar_groups:
            self.display_current_group()
        else:
            self.status_var.set("✓ All groups processed!")
            self.clear_display()
    
    def clear_display(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.photo_images.clear()
        self.selected_indices.clear()
        self.group_label_var.set("No groups remaining")
        self.selection_var.set("")
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.delete_selected_btn.config(state=tk.DISABLED)
        self.keep_largest_btn.config(state=tk.DISABLED)
        self.clear_selection_btn.config(state=tk.DISABLED)
    
    def prev_group(self):
        if self.current_group_idx > 0:
            self.current_group_idx -= 1
            self.display_current_group()
    
    def next_group(self):
        if self.current_group_idx < len(self.finder.similar_groups) - 1:
            self.current_group_idx += 1
            self.display_current_group()


def main():
    root = tk.Tk()
    app = ImageDedupeApp(root)
    root.mainloop()


if __name__ == "__main__":
    if not CLIP_AVAILABLE:
        print("\n⚠️  Warning: sentence-transformers not installed.")
        print("For better results, install it with:")
        print("pip install sentence-transformers\n")
    
    main()
