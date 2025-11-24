# Reza.naghdi (4022023047)

import re
import os
import sys
import json
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox

try:
    import PyPDF2
except ImportError:
    print("Error: PyPDF2 library not found. Please install it using 'pip install PyPDF2'.")
    sys.exit(1)



STOPWORDS = set([
    'a','an','the','and','or','not','in','on','at','for','to','of','is','are','was','were','be','by','with','as','that','this','these','those','it','can','will','have','from','may', 'all', 'do'
])
WORD_RE = re.compile(r"[A-Za-z0-9]+")

def standardize_case(text):
    """Normalized function: Converts text to lowercase."""
    return text.lower()

def split_words(text):
    """Tokenization function: Finds all words based on WORD_RE."""
    return WORD_RE.findall(standardize_case(text))




try:
    from nltk.stem.porter import PorterStemmer

    porter_stemmer = PorterStemmer()


    def reduce_word(w):
        return porter_stemmer.stem(w)


    print("Using NLTK Porter Stemmer (best quality)")
except ImportError:
    print("\nWarning: NLTK not found. Using simple fallback stemmer.")


    def simple_stem(w):
        suffixes = ['ing', 'ly', 'ed', 'ious', 'ive', 'es', 's', 'ment', 'able', 'er', 'est']
        for suf in sorted(suffixes, key=len, reverse=True):
            if w.endswith(suf) and len(w) > len(suf) + 3:
                return w[:-len(suf)]
        return w


    def reduce_word(w):
        return simple_stem(w)



class DictionaryNode:
    """Represents a node in the B-Tree dictionary."""
    def __init__(self, t, is_leaf=True):
        self.t = t
        self.keys = []
        self.children = []
        self.is_leaf = is_leaf

class TermDictionary:
    """Manages the B-Tree structure for term lookups."""
    def __init__(self, t=2):
        self.t = t
        self.root = DictionaryNode(t)

    def lookup_term(self, term_key, node=None):
        if node is None:
            node = self.root
        i = 0
        while i < len(node.keys) and term_key > node.keys[i]:
            i += 1
        if i < len(node.keys) and term_key == node.keys[i]:
            return True # Term found
        if node.is_leaf:
            return False
        return self.lookup_term(term_key, node.children[i])

    def _split_full_child(self, parent, child_idx):
        t = self.t
        node = parent.children[child_idx]
        new_node = DictionaryNode(t, node.is_leaf)
        parent.keys.insert(child_idx, node.keys[t-1])
        parent.children.insert(child_idx+1, new_node)
        new_node.keys = node.keys[t:]
        node.keys = node.keys[:t-1]
        if not node.is_leaf:
            new_node.children = node.children[t:]
            node.children = node.children[:t]

    def add_term(self, term_key):
        current_root = self.root
        if len(current_root.keys) == 2*self.t - 1:
            new_root = DictionaryNode(self.t, False)
            new_root.children.append(current_root)
            self._split_full_child(new_root, 0)
            self.root = new_root
            self._add_to_nonfull(new_root, term_key)
        else:
            self._add_to_nonfull(current_root, term_key)

    def _add_to_nonfull(self, node, term_key):
        if node.is_leaf:
            node.keys.append(term_key)
            node.keys.sort()
        else:
            i = len(node.keys) - 1
            while i >= 0 and term_key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2*self.t - 1:
                self._split_full_child(node, i)
                if term_key > node.keys[i]:
                    i += 1
            self._add_to_nonfull(node.children[i], term_key)
            
    def print_structure_console(self, node=None, level=0):
        if node is None:
            node = self.root
        prefix = '  ' * level + ('|-- ' if level > 0 else '')
        print(f"{prefix}Terms: {', '.join(node.keys)} {'(Leaf)' if node.is_leaf else ''}")
        for child in node.children:
            self.print_structure_console(child, level + 1)



document_directory = "./documents"
source_docs = {}

if not os.path.exists(document_directory):
    os.makedirs(document_directory)
    print(f"Created folder: {document_directory}. Please place 1.pdf, 2.pdf, and 3.pdf inside.")
    sys.exit(0)

print("--- Loading Source Documents (PDFs) ---")
for doc_key in range(1, 4):
    path = os.path.join(document_directory, f"{doc_key}.pdf")
    if not os.path.exists(path):
        print(f"Error: Document {doc_key}.pdf not found at {path}. Skipping.")
        continue

    try:
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            doc_content = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    doc_content += page_text + "\n"
            source_docs[doc_key] = doc_content
        print(f"Successfully loaded Doc {doc_key}.")
    except Exception as e:
        print(f"Error reading or processing {path}: {e}. Skipping.")
        continue

if not source_docs:
    print("\nNo documents were loaded. Exiting.")
    sys.exit(1)



term_to_postings = defaultdict(lambda: defaultdict(int))
dictionary_tree = TermDictionary(t=2)
indexing_log = []

def record_step(action, detail):
    """Function to record every step of the indexing process."""
    indexing_log.append({"step": action, "detail": detail})

print("\n--- Building Inverted Index and B-Tree Dictionary ---")

for doc_key, doc_content in source_docs.items():
    record_step("DOCUMENT_LOAD", f"Doc {doc_key}: {doc_content[:100].strip()}...")
    
    normalized_text = standardize_case(doc_content)
    record_step("NORMALIZE", f"Doc {doc_key}: {normalized_text[:100].strip()}...")
    
    tokens = split_words(normalized_text)
    record_step("TOKENIZE", f"Doc {doc_key}: {tokens[:20]}...")

    for token in tokens:
        if token in STOPWORDS:
            record_step("STOPWORD_REMOVE", f"Term '{token}' removed.")
            continue
            
        reduced_term = reduce_word(token)
        if not reduced_term: continue
        record_step("STEM_REDUCE", f"'{token}' -> '{reduced_term}'")
        
        term_to_postings[reduced_term][doc_key] += 1
        record_step("SID_UPDATE", f"Term '{reduced_term}' found in doc {doc_key} (count: {term_to_postings[reduced_term][doc_key]})")
        
        if not dictionary_tree.lookup_term(reduced_term):
            dictionary_tree.add_term(reduced_term)
            record_step("BTREE_INSERT", f"New term '{reduced_term}' added to B-Tree.")




def display_inverted_index_console(term_to_postings):
    """Prints the final inverted index to the console."""
    print("\n=== Standard Inverted Index (SID) Console Output) ===\n")
    for term in sorted(term_to_postings.keys()):
        postings = {int(k): v for k, v in term_to_postings[term].items()}
        print(f"{term:<15} -> {postings}")

display_inverted_index_console(term_to_postings)
print("\n=== Term Dictionary B-Tree Structure ===\n")
dictionary_tree.print_structure_console()

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/inverted_index_data.json', 'w', encoding='utf-8') as f:
    json.dump({k: dict(v) for k,v in term_to_postings.items()}, f, ensure_ascii=False, indent=2)
with open(f'{output_dir}/process_trace_log.json', 'w', encoding='utf-8') as f:
    json.dump(indexing_log, f, ensure_ascii=False, indent=2)
print(f"\nIndexing trace and SID data saved to {output_dir}/")



class IndexDisplayGUI:
    """Tkinter-based B-Tree Visualizer with custom aesthetics."""
    def __init__(self, btree, postings):
        self.btree   = btree
        self.postings = postings
        self.node_positions     = {}
        
        self.vertical_spacing = 120
        self.node_width_px = 120       
        self.node_height_px = 40       
        
        self.FILL_COLOR = '#f0fff0'  # Light Mint Green
        self.OUTLINE_COLOR = '#388e3c' # Dark Green 

        self.root = tk.Tk()
        self.root.title('Term Dictionary B-Tree Visualizer')
        self.canvas = tk.Canvas(
            self.root,
            width=1400, height=800,
            bg='white',
            scrollregion=(0,0,3000,2000)
        )
        hbar = tk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._calculate_width_unit(self.btree.root)
        self._set_node_placement(self.btree.root, 0)

        self._draw_structure()
        self.root.mainloop()


    def _calculate_width_unit(self, node):
        if node.is_leaf:
            self.node_positions[id(node)] = {'width': 1}
            return
        total = 0
        for child in node.children:
            self._calculate_width_unit(child)
            total += self.node_positions[id(child)]['width']
        self.node_positions[id(node)] = {'width': max(total, 1)}

    def _set_node_placement(self, node, start_x_unit):
        if node.is_leaf:
            self.node_positions[id(node)]['x'] = start_x_unit
            self.node_positions[id(node)]['d'] = self._determine_depth(node)
            return start_x_unit + 1

        current_x_unit = start_x_unit
        for child in node.children:
            current_x_unit = self._set_node_placement(child, current_x_unit)

        child_x_units = [self.node_positions[id(c)]['x'] for c in node.children]
        center_x_unit = (child_x_units[0] + child_x_units[-1]) / 2 if child_x_units else start_x_unit
        self.node_positions[id(node)]['x'] = center_x_unit
        self.node_positions[id(node)]['d'] = self._determine_depth(node)
        return max(current_x_unit, start_x_unit + 1)

    def _determine_depth(self, node):
        depth = 0
        current = node
        while hasattr(current, 'children') and current.children:
            current = current.children[0]
            depth += 1
        return depth
    # ----------------------------------------------------------
    
    def _draw_structure(self):
        self._draw_connections(self.btree.root)
        self._draw_term_nodes(self.btree.root)

    def _draw_connections(self, node):
        if not node.children:
            return
        px, py = self._get_pixel_coords(node)
        for child in node.children:
            cx, cy = self._get_pixel_coords(child)
            self.canvas.create_line(
                px, py + self.node_height_px//2,
                cx, cy - self.node_height_px//2,
                fill='gray', width=1
            )
            self._draw_connections(child)

    def _draw_term_nodes(self, node):
        x, y = self._get_pixel_coords(node)

        term_text = node.keys[0] if node.keys else 'EMPTY'
        if len(node.keys) > 1:
            term_text += f' (+{len(node.keys)-1})' 

        rect_id = self.canvas.create_rectangle(
            x-self.node_width_px//2, y-self.node_height_px//2,
            x+self.node_width_px//2, y+self.node_height_px//2,
            fill=self.FILL_COLOR, 
            outline=self.OUTLINE_COLOR, 
            width=2
        )
        text_id = self.canvas.create_text(
            x, y, text=term_text, font=('Consolas', 10, 'bold')
        )

        def on_click_show_postings(event, terms=node.keys):
            lines = ["--- Term Postings Data ---"]
            for term in terms:
                postings_data = dict(self.postings.get(term, {}))
                lines.append(f"{term:<10}: {postings_data}" if postings_data else f"{term:<10}: (No Postings)")
            messagebox.showinfo("Node Postings Data", "\n".join(lines) if lines else "No data")
            
        self.canvas.tag_bind(rect_id,   "<Button-1>", on_click_show_postings)
        self.canvas.tag_bind(text_id, "<Button-1>", on_click_show_postings)

        for child in node.children:
            self._draw_term_nodes(child)

    # ----------------------------------------------------------
    def _get_pixel_coords(self, node):
        """Converts logical (x_unit, depth) to screen (px, py)."""
        info = self.node_positions[id(node)]
        x_unit = info['x']
        depth = info['d']

        all_x_units = [p['x'] for p in self.node_positions.values()]
        max_x_unit = max(all_x_units) if all_x_units else 0
        
        canvas_width = max(self.canvas.winfo_width(), 1400)
        margin = 80 
        available_space = canvas_width - 2*margin
        
        px = margin + x_unit * (available_space / max(1, max_x_unit))
        py = 80 + depth * self.vertical_spacing
        return px, py



if source_docs:
    IndexDisplayGUI(dictionary_tree, term_to_postings)
