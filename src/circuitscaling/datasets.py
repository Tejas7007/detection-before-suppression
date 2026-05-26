#!/usr/bin/env python3
"""
IOI Dataset Generator — matching Wang et al. (2022) Appendix E.

Original paper: "Interpretability in the Wild" (arXiv:2211.00593)
  - ~100 single-token English first names
  - 15 ABBA + 15 BABA templates
  - 20 places, 20 objects (all single-token)
  - Symmetric generation (each prompt generates both ABBA and BABA versions)

Usage:
    from src.circuitscaling.datasets import IOIDataset
    dataset = IOIDataset(model, n_prompts=1000, seed=42)
    prompts = dataset.prompts       # list of str
    io_names = dataset.io_names     # list of str (indirect object)
    s_names = dataset.s_names       # list of str (subject = repeated name)
    io_ids = dataset.io_token_ids   # list of int (token id of IO)
    s_ids = dataset.s_token_ids     # list of int (token id of S)
"""
import random
from typing import List, Dict, Tuple, Optional

# ============================================================
# Templates from Wang et al. 2022, Appendix E (Table 7/Figure 14)
# ============================================================

# BABA: Subject (repeated name) appears first, IO appears second in intro
BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to",
]

# ABBA: IO appears first, Subject appears second in intro (but S is still repeated)
ABBA_TEMPLATES = [
    "Then, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [A] and [B] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [A] and [B] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [A] and [B] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [A] and [B] had a long argument, and afterwards [B] said to",
    "After [A] and [B] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [A] and [B] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [A] and [B] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [A] and [B] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [A] and [B] were commuting to the [PLACE], [B] gave a [OBJECT] to",
    "After the lunch, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Afterwards, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [A] and [B] had a long argument. Afterwards [B] said to",
    "The [PLACE] [A] and [B] went to had a [OBJECT]. [B] gave it to",
    "Friends [A] and [B] found a [OBJECT] at the [PLACE]. [B] gave it to",
]

ALL_TEMPLATES = BABA_TEMPLATES + ABBA_TEMPLATES

# ============================================================
# Candidate names — ~100 English first names
# At runtime we filter to single-token for the specific tokenizer.
# This list is deliberately large so that after filtering we still
# have ~50-80 usable names per model family.
# ============================================================

CANDIDATE_NAMES = [
    # Common short English names (high chance of single-token)
    "Aaron", "Adam", "Alan", "Alex", "Alice", "Amanda", "Amy", "Andrew",
    "Angela", "Anna", "Anne", "Arthur", "Ben", "Beth", "Bill", "Bob",
    "Brad", "Brian", "Carl", "Carol", "Charlie", "Chris", "Claire", "Colin",
    "Dan", "Daniel", "Dave", "David", "Dean", "Diana", "Don", "Donna",
    "Ed", "Edward", "Elena", "Ellen", "Emily", "Emma", "Eric", "Eva",
    "Frank", "Fred", "Gary", "George", "Glen", "Grace", "Greg", "Hannah",
    "Harry", "Helen", "Henry", "Holly", "Ian", "Iris", "Jack", "Jake",
    "James", "Jane", "Jason", "Jean", "Jeff", "Jennifer", "Jerry", "Jim",
    "Joan", "Joe", "John", "Jon", "Julia", "Julie", "Justin", "Karen",
    "Kate", "Keith", "Kelly", "Ken", "Kevin", "Kim", "Larry", "Laura",
    "Lee", "Leon", "Linda", "Lisa", "Louis", "Lucy", "Luke", "Lynn",
    "Marc", "Maria", "Marie", "Mark", "Martin", "Mary", "Matt", "Max",
    "Meg", "Michael", "Mike", "Nancy", "Neil", "Nick", "Noah", "Oliver",
    "Owen", "Pat", "Paul", "Peter", "Phil", "Rachel", "Ray", "Richard",
    "Rob", "Robert", "Robin", "Roger", "Ron", "Rose", "Roy", "Ruth",
    "Ryan", "Sam", "Sarah", "Scott", "Sean", "Sharon", "Simon", "Sophie",
    "Steve", "Susan", "Tim", "Tom", "Tony", "Victor", "Will", "Zoe",
]

# ============================================================
# Places and objects — 20 each, all single-token
# ============================================================

PLACES = [
    "store", "market", "garden", "museum", "library",
    "school", "church", "park", "beach", "lake",
    "forest", "river", "mountain", "castle", "theater",
    "restaurant", "hospital", "airport", "zoo", "gym",
]

OBJECTS = [
    "ring", "ball", "book", "bottle", "box",
    "card", "coin", "cup", "flower", "gift",
    "hat", "key", "lamp", "letter", "pen",
    "phone", "photo", "shirt", "shoe", "watch",
]


def filter_single_token_names(tokenizer, candidate_names: List[str] = None) -> List[str]:
    """Return only names that tokenize to a single token (with leading space)."""
    if candidate_names is None:
        candidate_names = CANDIDATE_NAMES
    
    single_token = []
    for name in candidate_names:
        # Names appear with a leading space in context: " Mary"
        tokens = tokenizer.encode(" " + name)
        # Some tokenizers prepend BOS — check both cases
        if len(tokens) == 1:
            single_token.append(name)
        elif len(tokens) == 2 and tokens[0] in (
            getattr(tokenizer, 'bos_token_id', None),
            getattr(tokenizer, 'pad_token_id', None),
        ):
            single_token.append(name)
    
    return single_token


def filter_single_token_words(tokenizer, words: List[str]) -> List[str]:
    """Return only words that tokenize to a single token (with leading space)."""
    single_token = []
    for w in words:
        tokens = tokenizer.encode(" " + w)
        if len(tokens) == 1:
            single_token.append(w)
        elif len(tokens) == 2 and tokens[0] in (
            getattr(tokenizer, 'bos_token_id', None),
            getattr(tokenizer, 'pad_token_id', None),
        ):
            single_token.append(w)
    return single_token


def gen_prompt_uniform(
    templates: List[str],
    names: List[str],
    places: List[str],
    objects: List[str],
    n: int,
    symmetric: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate IOI prompts following Wang et al. (2022) protocol.
    
    Each prompt is a dict with keys:
        text: the prompt string (ending before the IO name)
        IO: the indirect object name (correct answer)
        S: the subject name (distractor = repeated name)
        template_idx: which template was used
        pattern: "ABBA" or "BABA"
    
    If symmetric=True, each (A, B, template) also generates the swapped version.
    """
    rng = random.Random(seed)
    prompts = []
    
    while len(prompts) < n:
        template = rng.choice(templates)
        template_idx = templates.index(template)
        
        # Pick two distinct names
        name_A, name_B = rng.sample(names, 2)
        place = rng.choice(places)
        obj = rng.choice(objects)
        
        # Build prompt
        text = template
        text = text.replace("[A]", name_A)
        text = text.replace("[B]", name_B)
        text = text.replace("[PLACE]", place)
        text = text.replace("[OBJECT]", obj)
        
        # Determine pattern
        # In BABA templates, [B] comes first → B is repeated (S=B), A is IO
        # In ABBA templates, [A] comes first → B is repeated (S=B), A is IO
        # In both cases: A = IO (indirect object), B = S (subject, repeated)
        prompts.append({
            "text": text,
            "IO": name_A,  # The name that should be predicted
            "S": name_B,   # The repeated name (distractor)
            "template_idx": template_idx,
        })
        
        # Symmetric: also generate the swapped version
        if symmetric and len(prompts) < n:
            text2 = template
            text2 = text2.replace("[A]", name_B)
            text2 = text2.replace("[B]", name_A)
            text2 = text2.replace("[PLACE]", place)
            text2 = text2.replace("[OBJECT]", obj)
            
            prompts.append({
                "text": text2,
                "IO": name_B,
                "S": name_A,
                "template_idx": template_idx,
            })
    
    return prompts[:n]


class IOIDataset:
    """
    Full-scale IOI dataset matching Wang et al. (2022).
    
    Usage:
        model = HookedTransformer.from_pretrained("gpt2-medium")
        dataset = IOIDataset(model, n_prompts=1000)
        tokens = model.to_tokens(dataset.prompts, prepend_bos=True)
    """
    
    def __init__(
        self,
        model,
        n_prompts: int = 1000,
        templates: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        symmetric: bool = True,
        seed: int = 42,
    ):
        tokenizer = model.tokenizer
        
        # Filter names/places/objects to single-token
        if names is None:
            self.names = filter_single_token_names(tokenizer)
        else:
            self.names = filter_single_token_names(tokenizer, names)
        
        if places is None:
            self.places = filter_single_token_words(tokenizer, PLACES)
        else:
            self.places = filter_single_token_words(tokenizer, places)
        
        if objects is None:
            self.objects = filter_single_token_words(tokenizer, OBJECTS)
        else:
            self.objects = filter_single_token_words(tokenizer, objects)
        
        if templates is None:
            templates = ALL_TEMPLATES
        
        assert len(self.names) >= 2, (
            f"Need at least 2 single-token names, got {len(self.names)}: {self.names}"
        )
        
        # Generate prompts
        raw = gen_prompt_uniform(
            templates, self.names, self.places, self.objects,
            n_prompts, symmetric, seed
        )
        
        self.raw_prompts = raw
        self.prompts = [p["text"] for p in raw]
        self.io_names = [p["IO"] for p in raw]
        self.s_names = [p["S"] for p in raw]
        
        # Get token IDs
        self.io_token_ids = [
            tokenizer.encode(" " + name)[-1] for name in self.io_names
        ]
        self.s_token_ids = [
            tokenizer.encode(" " + name)[-1] for name in self.s_names
        ]
        
        self.n_prompts = len(self.prompts)
        self.n_names = len(self.names)
        self.n_templates = len(templates)
        self.tokenizer = tokenizer
    
    def summary(self) -> str:
        return (
            f"IOIDataset: {self.n_prompts} prompts, "
            f"{self.n_names} single-token names, "
            f"{self.n_templates} templates, "
            f"{len(self.places)} places, {len(self.objects)} objects"
        )


# ============================================================
# Legacy interface (backward compatible with old 15-name version)
# ============================================================

LEGACY_NAMES = [
    "John", "Mary", "Tom", "Kate", "Raj", "Liam", "Mia", "Omar",
    "Lina", "Chen", "Sara", "Ivy", "Alex", "Noah", "Zoe",
]

LEGACY_TEMPLATES = [
    "When {A} and {B} went to the store, {A} gave a gift to",
]


def make_ioi_prompts_legacy(n: int, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Legacy interface: 15 names, 1 template. Returns (prompts, io_names, s_names)."""
    rng = random.Random(seed)
    prompts, ios, subjs = [], [], []
    for _ in range(n):
        A, B = rng.sample(LEGACY_NAMES, 2)
        prompts.append(f"When {A} and {B} went to the store, {A} gave a gift to")
        ios.append(B)
        subjs.append(A)
    return prompts, ios, subjs


if __name__ == "__main__":
    # Quick test: show how many names survive per model
    from transformer_lens import HookedTransformer
    
    for model_name in ["gpt2", "gpt2-medium", "EleutherAI/pythia-410m-deduped"]:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        dataset = IOIDataset(model, n_prompts=20, seed=42)
        print(f"  {dataset.summary()}")
        print(f"  Names: {dataset.names[:10]}...")
        print(f"  Sample prompt: {dataset.prompts[0]}")
        print(f"  IO={dataset.io_names[0]}, S={dataset.s_names[0]}")
        del model
