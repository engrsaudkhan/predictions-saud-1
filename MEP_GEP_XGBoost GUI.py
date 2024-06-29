import tkinter as tk
from tkinter import ttk
from math import pow, sqrt
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
class RangeInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Graphical User Interface (GUI) for compressive strength prediction of metakaolin-based cement mortars")
        master.configure(background="#f0f0f0")
        window_width = 760
        window_height = 800
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_cord = 0
        y_cord = 0
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        x_cord = 0
        y_cord = 0
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        main_heading = tk.Label(master, text="Graphical User Interface (GUI) for: \n Compressive strength prediction of metakaolin-based cement mortars",
                                bg="#444444", fg="#FFFFFF", font=("Helvetica", 16, "bold"), pady=10)
        main_heading.pack(side=tk.TOP, fill=tk.X)
        self.content_frame = tk.Frame(master, bg="#E8E8E8")
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=50, anchor=tk.CENTER)
        self.canvas = tk.Canvas(self.content_frame, bg="#E8E8E8")
        self.scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#E8E8E8")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.input_frame.pack(side=tk.TOP, fill="both", padx=10, pady=10, expand=True)
        heading = tk.Label(self.input_frame, text="Input Parameters", bg="#FFFFFF", font=("Helvetica", 16, "bold"), padx=10, pady=10)
        heading.grid(row=0, column=0, columnspan=2, pady=10)        
        self.output_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.output_frame.pack(side=tk.TOP, fill="both", padx=20, pady=20)
        heading = tk.Label(self.output_frame, text="Predictions", bg="#FFFFFF", fg="black", font=("Helvetica", 16, "bold"), pady=10)
        heading.grid(row=0, column=0, columnspan=2, pady=10)
        self.G3C8 = 0.530799907502179
        self.G3C5 = -0.539689524535934
        self.G4C1 = 7.2288131524787
        self.create_entry("Age of specimen:", 90.0, 1)
        self.create_entry("Max. Diameter of Aggregate:", 2.0, 3)
        self.create_entry("Metakaolin percentage in relation to total binder content:", 0.0, 5)
        self.create_entry("Water-to-binder ratio:", 0.49, 7)
        self.create_entry("Binder-to-sand ratio:", 0.36, 9)
        self.G1C7 = 10.8342013254869
        self.G1C5 = 3.5001410694929
        self.G1C9 = 5.23934169605863
        self.G2C0 = 7.04505569473521
        self.G2C9 = -5.79039621290785
        self.G2C4 = -2.02649451434779
        self.G2C7 = -1.43861287640308
        self.G2C6 = -5.67222960443133
        self.G3C1 = 9.14994426494925
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.calculate_button_a = tk.Button(self.output_frame, text="Gene Expression Programming (GEP)", command=self.calculate_y_a,
                                          bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_a.grid(row=1, column=0, pady=10, padx=10)
        self.a_output_text_a = tk.Text(self.output_frame, height=2, width=30)
        self.a_output_text_a.grid(row=1, column=1, padx=10, pady=10)
        self.calculate_button_b = tk.Button(self.output_frame, text="Multi Expression Programming (MEP)", command=self.calculate_y_b,
                                            bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_b.grid(row=2, column=0, pady=10, padx=10)
        self.G4C8 = 1.68016050020444
        self.G4C5 = 6.95993912486141
        self.G5C6 = -12.1084164439053
        self.G5C5 = -2.18836798631613
        self.G5C3 = 0.585190328622753
        self.b_output_text_b = tk.Text(self.output_frame, height=2, width=30)
        self.b_output_text_b.grid(row=2, column=1, padx=10, pady=10)
        self.c_button_c = tk.Button(self.output_frame, text="Extreme Gradient Boosting (XGB)", command=self.calculate_c_c,
                                        bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.c_button_c.grid(row=4, column=0, pady=10, padx=10)
        self.c_output_text_c = tk.Text(self.output_frame, height=2, width=30)
        self.c_output_text_c.grid(row=4, column=1, padx=10, pady=10)
        developer_info = tk.Label(text="This GUI is developed by combined efforts of:\nMuhammad Saud Khan (khans28@myumanitoba.ca), University of Manitoba, Canada\nZohaib Mehmood (zoohaibmehmood@gmail.com), COMSATS University Islamabad, Pakistan",
                                  bg="light blue", fg="purple", font=("Helvetica", 11, "bold"), pady=10)
        developer_info.pack()
    def create_entry(self, text, default_val, row):
        label = tk.Label(self.input_frame, text=text, font=("Helvetica", 12, "bold italic"), fg="darkred", bg="white", anchor="w")
        label.grid(row=row*2, column=0, padx=10, pady=5, sticky="w")
        entry = tk.Entry(self.input_frame, font=("Helvetica", 12), fg="darkgreen", bg="white", width=15, bd=2, relief=tk.GROOVE)
        entry.insert(0, f"{default_val:.2f}")
        entry.grid(row=row*2, column=1, padx=10, pady=5, sticky="w")
        setattr(self, f'entry_{row}', entry)
    def get_entry_values(self):
        try:
            d1 = float(self.entry_1.get())
            d2 = float(self.entry_3.get())
            d3 = float(self.entry_5.get())
            d4 = float(self.entry_7.get())
            d5 = float(self.entry_9.get())
            return d1, d2, d3, d4, d5
        except ValueError as ve:
            return None
    def calculate_y_a(self):
        values = self.get_entry_values()
        if values is None:
            self.a_output_text_a.delete(1.0, tk.END)
            self.a_output_text_a.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        y = 0
        y += (d2 * math.log((self.G1C7 / pow((self.G1C5 + math.log(sqrt((pow(pow((d5 * math.log(d2)), 2.0), 2.0) / pow(sqrt((sqrt(d5) * self.G1C9)), 2.0))))), 2.0))))
        y += sqrt(pow((math.log(pow((((((self.G2C7 * d2) /pow(self.G2C6, 2.0)) * pow(self.G2C9, 2.0)) +(self.G2C0 - pow(self.G2C4, 2.0))) / d4), 2.0)) /pow(sqrt(pow(d4, 2.0)), 2.0)), 2.0))
        y += math.log((d3 +((d5 /(d5 -d2)) /((((d4 * self.G3C1) * pow(self.G3C8, 2.0)) + math.log(pow(d5, 2.0))) /pow(((d5 /self.G3C5) /(d5 -d2)), 2.0)))))
        y += (self.G4C1 * sqrt(sqrt((d1 * pow(((((((self.G4C8 / d5) / (d1 - self.G4C8)) * ((d5 - d2) /(d4 - self.G4C5))) + d4) - d1) / d1), 2.0)))))
        y += (d4 * (pow(((sqrt((self.G5C3 /(d5 / sqrt(math.log(((d1 + self.G5C3) /d1)))))) /d5) + (d2 / self.G5C5)), 2.0) * self.G5C6))
        self.a_output_text_a.delete(1.0, tk.END)
        self.a_output_text_a.insert(tk.END, f"{y:.2f}")
        self.a_output_text_a.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
    def calculate_y_b(self):
        values = self.get_entry_values()
        if values is None:
            self.xgboost_output_text_b.delete(1.0, tk.END)
            self.xgboost_output_text_b.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        prg = [0] * 100
        prg[0] = d4
        prg[1] = d2
        prg[2] = d3
        prg[3] = d5
        prg[4] = math.sqrt(prg[0])
        prg[5] = d5
        prg[6] = prg[5] * prg[1]
        prg[7] = prg[6] - prg[0]
        prg[8] = prg[5] * prg[0]
        prg[9] = prg[0] / prg[0]
        prg[10] = prg[9] + prg[8]
        prg[11] = prg[6] * prg[6]
        prg[12] = prg[1] + prg[6]
        prg[13] = prg[5] / prg[0]
        prg[14] = prg[2] + prg[13]
        prg[15] = prg[0] + prg[9]
        prg[16] = d4
        prg[17] = d1
        prg[18] = prg[10] / prg[7]
        prg[19] = prg[15] * prg[15]
        prg[20] = d5
        prg[21] = prg[12] / prg[14]
        prg[22] = prg[15] + prg[15]
        prg[23] = prg[18] * prg[18]
        prg[24] = d3
        prg[25] = d1
        prg[26] = prg[25] - prg[5]
        prg[27] = math.sqrt(prg[26])
        prg[28] = prg[26] - prg[22]
        prg[29] = prg[23] * prg[23]
        prg[30] = prg[0] + prg[24]
        prg[31] = prg[28] * prg[28]
        prg[32] = prg[27] * prg[9]
        prg[33] = prg[6] * prg[22]
        prg[34] = prg[33] - prg[9]
        prg[35] = prg[33] * prg[27]
        prg[36] = d5
        prg[37] = prg[34] - prg[18]
        prg[38] = prg[18] - prg[29]
        prg[39] = prg[34] / prg[11]
        prg[40] = prg[18] / prg[25]
        prg[41] = prg[15] / prg[28]
        prg[42] = prg[27] - prg[21]
        prg[43] = prg[30] / prg[30]
        prg[44] = prg[13] * prg[13]
        prg[45] = prg[37] + prg[33]
        prg[46] = prg[16] + prg[45]
        prg[47] = d2
        prg[48] = prg[6] * prg[6]
        prg[49] = prg[39] - prg[38]
        prg[50] = prg[14] * prg[14]
        prg[51] = d3
        prg[52] = d4
        prg[53] = prg[41] + prg[5]
        prg[54] = prg[32] / prg[49]
        prg[55] = prg[23] + prg[9]
        prg[56] = d3
        prg[57] = math.sqrt(prg[43])
        prg[58] = prg[20] * prg[20]
        prg[59] = prg[22] / prg[0]
        prg[60] = d5
        prg[61] = d2
        prg[62] = prg[53] + prg[9]
        prg[63] = prg[39] / prg[53]
        prg[64] = prg[62] * prg[33]
        prg[65] = d2
        prg[66] = prg[57] + prg[61]
        prg[67] = prg[39] + prg[15]
        prg[68] = prg[59] * prg[59]
        prg[69] = prg[42] * prg[42]
        prg[70] = prg[32] - prg[54]
        prg[71] = d1
        prg[72] = prg[69] * prg[69]
        prg[73] = prg[45] - prg[50]
        prg[74] = prg[63] + prg[62]
        prg[75] = d5
        prg[76] = math.sqrt(prg[65])
        prg[77] = prg[18] + prg[42]
        prg[78] = prg[68] + prg[70]
        prg[79] = prg[74] + prg[78]
        prg[80] = prg[59] + prg[66]
        prg[81] = d2
        prg[82] = prg[48] / prg[15]
        prg[83] = d5
        prg[84] = prg[79] - prg[45]
        prg[85] = d5
        prg[86] = d1
        prg[87] = d3
        prg[88] = prg[84] + prg[42]
        prg[89] = prg[48] + prg[79]
        prg[90] = d2
        prg[91] = prg[27] * prg[27]
        prg[92] = prg[7] * prg[7]
        prg[93] = prg[88] - prg[40]
        prg[94] = prg[50] * prg[42]
        prg[95] = d1
        prg[96] = prg[53] - prg[2]
        prg[97] = prg[75] * prg[49]
        prg[98] = d3
        prg[99] = prg[64] * prg[54]        
        outputs = [0]   
        outputs[0] = prg[93]       
        self.b_output_text_b.delete(1.0, tk.END)
        self.b_output_text_b.insert(tk.END, f"{outputs[0]:.2f}")
        self.b_output_text_b.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
        return outputs
    def calculate_c_c(self):
        values = self.get_entry_values()
        if values is None:
            self.c_output_text_c.delete(1.0, tk.END)
            self.c_output_text_c.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        try:
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\software-and-prediction-main\MEP_GEP_XGBoost GUI"
            filename = r"MK.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=500)
            regressor= MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=40,
                reg_lambda=0.01,
                gamma=1,
                max_depth=5
            ))
            model=regressor.fit(x_train, y_train)
            model= model.fit(x, y)
            y_pred = model.predict(x_train)
            input_data = np.array([d1, d2, d3, d4, d5]).reshape(1, -1)
            y_pred = model.predict(input_data)
            self.c_output_text_c.delete(1.0, tk.END)
            self.c_output_text_c.insert(tk.END, f"{y_pred[0][0]:.2f}")
            self.c_output_text_c.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
        except FileNotFoundError:
            self.c_output_text_c.delete(1.0, tk.END)
            self.c_output_text_c.insert(tk.END, "Error: Excel file not found")
        except ValueError as ve:
            self.c_output_text_c.delete(1.0, tk.END)
            self.c_output_text_c.insert(tk.END, "Error: Invalid data format")
        except Exception as e:
            self.c_output_text_c.delete(1.0, tk.END)
            self.c_output_text_c.insert(tk.END, "Error: XGBoost Density prediction failed")
if __name__ == "__main__":
    root = tk.Tk()
    gui = RangeInputGUI(root)
    root.mainloop()