import subprocess

def run_rnafold(sequence):
    """
    System call to RNAfold from the ViennaRNA package.
    Calculates the minimum free energy (MFE) secondary structure.
    """
    try:
        process = subprocess.Popen(['RNAfold', '--noPS'], 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        stdout, stderr = process.communicate(input=sequence)
        
        # Parsing logic for MFE value
        # Example output: "(((.((....)).))) (-5.20)"
        lines = stdout.splitlines()
        if len(lines) >= 2:
            mfe_line = lines[1]
            mfe = float(mfe_line.split('(')[-1].replace(')', '').strip())
            return mfe
    except Exception as e:
        print(f"Error running RNAfold: {e}")
        return None

if __name__ == "__main__":
    test_seq = "GGCGAAAAGGCCUACGAUCC"
    mfe = run_rnafold(test_seq)
    print(f"Sequence: {test_seq}")
    print(f"MFE: {mfe} kcal/mol")
