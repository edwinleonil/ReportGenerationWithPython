import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from utils_reportgen import generate_report
from docx2pdf import convert
import os
import yaml

# TODO: correct classified images needs to be added in the report
# TODO: incorrect classified images needs to be added in the report
class Application(tk.Frame):
    def __init__(self, window=None):
        super().__init__(window)
        self.window = window
        self.window.title("ReportGenApp") # set the window title
        self.window.resizable(width=False, height=False)
        self.config_data = self.load_config()  # Load the config data
        self.create_widgets()

    def load_config(self):
        # Load the config data from the yaml file
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def create_widgets(self):
        
        tk.Label(self, text="Word and PDF Report Generator", font=("Arial", 16)).grid(row=0, column=1, columnspan=3, padx=10, pady=10)

        self.select_training_folder_button = tk.Button(self, text="Training images folder path", command=self.select_training_folder)
        self.select_training_folder_button.grid(row=2, column=0, sticky="w", padx=10, pady=10)

        self.training_folder_path_entry = tk.Entry(self, width=150)
        self.training_folder_path_entry.insert(0, self.config_data['training_images_path'])
        self.training_folder_path = self.config_data['training_images_path']
        self.training_folder_path_entry.grid(row=2, column=2, sticky="w", padx=10, pady=10)

        self.select_testing_folder_button = tk.Button(self, text="Testing images folder path", command=self.select_testing_folder)
        self.select_testing_folder_button.grid(row=5, column=0, sticky="w", padx=10, pady=10)

        self.testing_folder_path_entry = tk.Entry(self, width=150)
        self.testing_folder_path_entry.insert(0, self.config_data['testing_images_path']) 
        self.testing_folder_path = self.config_data['testing_images_path']
        self.testing_folder_path_entry.grid(row=5, column=2, sticky="w", padx=10, pady=10)

        self.select_probability_file_button = tk.Button(self, text="Model prediction probability file path", command=self.select_probability_files)
        self.select_probability_file_button.grid(row=7, column=0, sticky="w", padx=10, pady=10)

        self.probability_file_path_entry = tk.Entry(self, width=150)
        self.probability_file_path_entry.insert(0, self.config_data['probabilities_file_paths']) 
        self.probability_file_paths = self.config_data['probabilities_file_paths']
        self.probability_file_path_entry.grid(row=7, column=2, sticky="w",  padx=10, pady=10)

        self.select_output_folder_button = tk.Button(self, text="Output folder path", command=self.select_output_folder)
        self.select_output_folder_button.grid(row=8, column=0, sticky="w", padx=10, pady=10)

        self.output_folder_path_entry = tk.Entry(self, width=150)
        self.output_folder_path_entry.insert(0, self.config_data['output_path'])
        self.output_folder_path = self.config_data['output_path']
        self.output_folder_path_entry.grid(row=8, column=2, sticky="w", padx=10, pady=10)

        self.generate_report_button = tk.Button(self, text="Generate Report", command=self.generate_report, bg="gray", fg="white")
        self.generate_report_button.grid(row=9, column=0, columnspan=3, sticky="we", padx=10, pady=10)

    def generate_report(self):
        num_of_empty_entries = 0
        # Check if the training_folder_path_entry value is empty
        if self.training_folder_path_entry.get() == "":
            # Show an error message box
            messagebox.showerror("Error", "Please select the training images folder path.")
            num_of_empty_entries += 1
            return
        
        # Check if the testing_folder_path_entry value is empty
        if self.testing_folder_path_entry.get() == "":
            # Show an error message box
            messagebox.showerror("Error", "Please select the testing images folder path.")
            num_of_empty_entries += 1
            return
        
        # Check if the probability_file_path_entry value is empty
        if self.probability_file_path_entry.get() == "":
            # Show an error message box
            messagebox.showerror("Error", "Please select the model prediction probability file path.")
            num_of_empty_entries += 1
            return
        
        # Check if the output_folder_path_entry value is empty
        if self.output_folder_path_entry.get() == "":
            # Show an error message box
            messagebox.showerror("Error", "Please select the output folder path.")
            num_of_empty_entries += 1
            return

        if num_of_empty_entries == 0:

            # get the list of probability file paths
            self.probability_file_paths = self.probability_file_path_entry.get().split(', ')
            
            for prob_file_path in self.probability_file_paths:
                # Call the generate_report function
                wordOutputPath = generate_report(self.training_folder_path, self.testing_folder_path, prob_file_path, self.output_folder_path)

                # Convert the report to PDF
                report_path = os.path.abspath(wordOutputPath)
                # convert the pdf to a word document
                pdf_path = report_path[:-4]+'pdf'
                # convert the word document to a pdf
                convert(report_path, pdf_path)

                # delete the word file
                os.remove(report_path)

            # print a message to indicate that all reports have been generated
            messagebox.showinfo("All Reports Generated", "All reports have been generated and saved as PDFs.")
            # quit the application
            self.window.quit()
    
    def select_training_folder(self):
        # Show a file dialog to select the training images folder path
        self.training_folder_path = filedialog.askdirectory()

        # Update the training folder path entry with the selected folder path
        self.training_folder_path_entry.delete(0, tk.END)
        self.training_folder_path_entry.insert(0, self.training_folder_path)

        # save this path in the config file
        self.config_data['training_images_path'] = self.training_folder_path
        with open('config.yaml', 'w') as file:
            yaml.dump(self.config_data, file)
        return self.training_folder_path

    def select_testing_folder(self):
        # Show a file dialog to select the testing images folder path
        self.testing_folder_path = filedialog.askdirectory()

        # Update the testing folder path entry with the selected folder path
        self.testing_folder_path_entry.delete(0, tk.END)
        self.testing_folder_path_entry.insert(0, self.testing_folder_path)

        # save this path in the config file
        self.config_data['testing_images_path'] = self.testing_folder_path
        with open('config.yaml', 'w') as file:
            yaml.dump(self.config_data, file)
        return self.testing_folder_path

    def select_probability_files(self):
        # Show a file dialog to select the probability file paths
        self.probability_file_paths = list(filedialog.askopenfilenames())

        # Update the probability file path entry with the selected file paths
        self.probability_file_path_entry.delete(0, tk.END)
        self.probability_file_path_entry.insert(0, ', '.join(self.probability_file_paths))

        # save these paths in the config file
        self.config_data['probabilities_file_paths'] = self.probability_file_paths
        with open('config.yaml', 'w') as file:
            yaml.dump(self.config_data, file)
        return self.probability_file_paths

    def select_output_folder(self):
        # Show a file dialog to select the output folder path
        self.output_folder_path = filedialog.askdirectory()

        # Update the output folder path entry with the selected folder path
        self.output_folder_path_entry.delete(0, tk.END)
        self.output_folder_path_entry.insert(0, self.output_folder_path)

        # save this path in the config file
        self.config_data['output_path'] = self.output_folder_path
        with open('config.yaml', 'w') as file:
            yaml.dump(self.config_data, file)
        return self.output_folder_path

window = tk.Tk()
app = Application(window=window)
app.grid(sticky="nsew")
window.mainloop()