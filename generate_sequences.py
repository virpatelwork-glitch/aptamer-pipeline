import random

# Fixed random seed for reproducibility
random.seed(42)

def generate_l1_sequence(length=20):
    """
    Level 1: Baseline random generation with GC-content filtering.
    Ensures sequences are not biased by specific motifs.
    """
    nts = ['A', 'U', 'G', 'C']
    while True:
        seq = ''.join(random.choices(nts, k=length))
        gc = (seq.count('G') + seq.count('C')) / length
        if 0.4 <= gc <= 0.6:
            return seq

def generate_l2_sequence(motif='GAAA'):
    """
    Level 2: Motif-guided design.
    Inserts stable tetraloops (GAAA or UUCG) to promote structural stability.
    """
    nts = ['A', 'U', 'G', 'C']
    stem = ''.join(random.choices(nts, k=4))
    comp = {'A':'U','U':'A','G':'C','C':'G'}
    stem_rev = ''.join([comp[n] for n in reversed(stem)])
    return stem + motif + stem_rev + ''.join(random.choices(nts, k=8))

def mutate_sequence(seq, num_mutations=1):
    """
    Iterative mutation logic for Level 4 refinement.
    Simulates the process of fine-tuning sequences for better binding.
    """
    nts = ['A', 'U', 'G', 'C']
    seq_list = list(seq)
    for _ in range(num_mutations):
        idx = random.randint(0, len(seq)-1)
        seq_list[idx] = random.choice(nts)
    return "".join(seq_list)

if __name__ == "__main__":
    print("Generating example sequences...")
    print(f"L1: {generate_l1_sequence()}")
    print(f"L2: {generate_l2_sequence()}")
    print(f"L4 (mutated L2): {mutate_sequence(generate_l2_sequence())}")
