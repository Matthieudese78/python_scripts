# %%
import jupytext
import sys

# %%
if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = input_file.rsplit(".", 1)[0] + ".ipynb"
    print(f"Converting into output file {output_file}")
    jupytext.write(jupytext.read(input_file), output_file)

# %%
