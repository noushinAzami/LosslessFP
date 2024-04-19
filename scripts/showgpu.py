import csv
import matplotlib.pyplot as plt

input_file = 'gpu_output_compress.csv'

# Read the CSV file
x_values = []
y_values = []
modes = []

with open(input_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x_values.append(float(row['Comp']))
        y_values.append(float(row['Ratio']))
        modes.append('SP' + row['Mode'])

# Create scatter plot
plt.scatter(x_values, y_values)

# Add labels and title
plt.xlabel('Compression Throughput (GB/s)')
plt.ylabel('Compression Ratio')
plt.title('Single gpu - Compression')

# Add legend
for mode, x, y in zip(modes, x_values, y_values):
    plt.text(x, y, mode)

# Set the number of digits after the decimal point
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# Save the plot as PNG file
plt.savefig('single-gpu-comp.png')

# Reset variables for decompression data
input_file = 'gpu_output_decompress.csv'

# Read the decompression CSV file
dx_values = []
dy_values = []
dmodes = []

with open(input_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dx_values.append(float(row['Decomp']))
        dy_values.append(float(row['Ratio']))
        dmodes.append('SP' + row['Mode'])

# Create scatter plot for decompression data
plt.figure()  # Create a new figure
plt.scatter(dx_values, dy_values)

# Add labels and title for decompression plot
plt.xlabel('Decompression Throughput (GB/s)')
plt.ylabel('Decompression Ratio')
plt.title('Single gpu - Decompression')

# Add legend for decompression plot
for mode, x, y in zip(dmodes, dx_values, dy_values):
    plt.text(x, y, mode)

# Set the number of digits after the decimal point for decompression plot
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# Save the decompression plot as PNG file
plt.savefig('single-gpu-decomp.png')

# Show both plots
plt.show()

