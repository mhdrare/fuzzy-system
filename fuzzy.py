import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import os
import time

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.data_quality_report = {}
    
    def load_and_validate_data(self, source_path='./source/'):
        print("TAHAP 1: PREPROCESSING DAN VALIDASI DATA")
        print("="*60)
        
        file_paths = {
            'gk': os.path.join(source_path, 'gk.df.csv'),
            'upah': os.path.join(source_path, 'upah.df.csv'),
            'peng': os.path.join(source_path, 'peng.df.csv'),
            'ump': os.path.join(source_path, 'ump.df.csv')
        }
        
        datasets = {}
        
        # Load datasets
        for name, path in file_paths.items():
            try:
                df = pd.read_csv(path)
                datasets[name] = df
                print(f"{name.upper()}: {df.shape}")
            except FileNotFoundError:
                print(f"{name.upper()}: File tidak ditemukan - {path}")
                return None, None
        
        # Validasi kualitas data
        quality_report = self.validate_data_quality(datasets)
        processed_data = self.process_and_integrate_data(datasets)
        
        return processed_data, quality_report
    
    def validate_data_quality(self, datasets):
        print("\nVALIDASI KUALITAS DATA:")
        print("-" * 40)
        
        quality_report = {}
        
        for name, df in datasets.items():
            report = {
                'total_rows': len(df),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Deteksi outliers untuk kolom numerik
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            
            report['outliers'] = outliers
            quality_report[name] = report
            
            print(f"\n{name.upper()}:")
            print(f"  Total baris: {report['total_rows']}")
            print(f"  Missing values: {report['missing_values']} ({report['missing_percentage']:.2f}%)")
            print(f"  Duplicate rows: {report['duplicate_rows']}")
            if outliers:
                total_outliers = sum(outliers.values())
                print(f"  Total outliers: {total_outliers}")
        
        self.data_quality_report = quality_report
        return quality_report
    
    def handle_missing_values(self, df, strategy='mean'):
        if strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'forward_fill':
            df.fillna(method='ffill', inplace=True)
        
        return df
    
    def remove_outliers(self, df, columns, method='iqr'):
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def process_and_integrate_data(self, datasets):
        print("\nINTEGRASI DATA:")
        print("-" * 20)
        
        upah_df = datasets['upah']
        ump_df = datasets['ump']
        gk_df = datasets['gk']
        peng_df = datasets['peng']
        
        processed_data = []
        
        # Ambil tahun terbaru
        available_years = sorted(upah_df['tahun'].unique())
        years_to_process = available_years[-3:] if len(available_years) >= 3 else available_years
        provinces = sorted(upah_df['provinsi'].unique())
        
        print(f"Tahun diproses: {years_to_process}")
        print(f"Provinsi: {len(provinces)} provinsi")
        
        for year in years_to_process:
            for province in provinces:
                try:
                    upah_filtered = upah_df[
                        (upah_df['tahun'] == year) & 
                        (upah_df['provinsi'] == province)
                    ]
                    
                    ump_filtered = ump_df[
                        (ump_df['tahun'] == year) & 
                        (ump_df['provinsi'] == province)
                    ]
                    
                    gk_total = gk_df[
                        (gk_df['tahun'] == year) & 
                        (gk_df['provinsi'] == province)
                    ].groupby(['tahun', 'provinsi'])['gk'].sum().reset_index()
                    
                    peng_total = peng_df[
                        (peng_df['tahun'] == year) & 
                        (peng_df['provinsi'] == province)
                    ].groupby(['tahun', 'provinsi'])['peng'].sum().reset_index()
                    
                    if all(len(df) > 0 for df in [upah_filtered, ump_filtered, gk_total, peng_total]):
                        upah_value = upah_filtered['upah'].iloc[0]
                        ump_value = ump_filtered['ump'].iloc[0]
                        gk_value = gk_total['gk'].iloc[0]
                        peng_value = peng_total['peng'].iloc[0]
                        
                        if all(pd.notnull([upah_value, ump_value, gk_value, peng_value])) and \
                            all(val > 0 for val in [upah_value, ump_value, gk_value, peng_value]):
                            
                            processed_data.append({
                                'tahun': year,
                                'provinsi': province,
                                'upah_per_jam': float(upah_value),
                                'ump': float(ump_value),
                                'garis_kemiskinan': float(gk_value),
                                'pengeluaran_perkapita': float(peng_value)
                            })
                            
                except Exception as e:
                    continue
        
        if processed_data:
            result_df = pd.DataFrame(processed_data)
            # Handle missing values dan outliers
            result_df = self.handle_missing_values(result_df)
            
            print(f"Data terintegrasi: {result_df.shape}")
            return result_df
        else:
            print("Tidak ada data yang berhasil diproses")
            return None
        
class MamdaniFuzzySystem:
    def __init__(self):
        self.name = "Mamdani"
        
    def create_membership_functions(self):
        # Fungsi membership manual untuk Mamdani
        self.input_ranges = {
            'ratio_upah_ump': np.arange(0, 3, 0.01),
            'ratio_upah_gk': np.arange(0, 15, 0.1),
            'efisiensi_pengeluaran': np.arange(0, 1.2, 0.01)
        }
        
        self.output_range = np.arange(0, 101, 1)
        
        # Membership functions untuk input
        self.membership_functions = {
            'ratio_upah_ump': {
                'rendah': lambda x: np.maximum(0, np.minimum(1, (1.0 - x) / 0.4)),
                'sedang': lambda x: np.maximum(0, np.minimum((x - 0.6) / 0.4, (1.5 - x) / 0.5)),
                'tinggi': lambda x: np.maximum(0, np.minimum(1, (x - 1.2) / 0.3))
            },
            'ratio_upah_gk': {
                'sangat_rendah': lambda x: np.maximum(0, np.minimum(1, (3 - x) / 3)),
                'rendah': lambda x: np.maximum(0, np.minimum((x - 2) / 1.5, (5 - x) / 1.5)),
                'sedang': lambda x: np.maximum(0, np.minimum((x - 4) / 2, (8 - x) / 2)),
                'tinggi': lambda x: np.maximum(0, np.minimum((x - 7) / 2, (12 - x) / 3)),
                'sangat_tinggi': lambda x: np.maximum(0, np.minimum(1, (x - 10) / 2))
            },
            'efisiensi_pengeluaran': {
                'boros': lambda x: np.maximum(0, np.minimum(1, (0.7 - x) / 0.2)),
                'normal': lambda x: np.maximum(0, np.minimum((x - 0.5) / 0.2, (0.9 - x) / 0.2)),
                'hemat': lambda x: np.maximum(0, np.minimum(1, (x - 0.8) / 0.1))
            }
        }
        
        # Output membership functions
        self.output_membership = {
            'sangat_rendah': lambda x: np.maximum(0, np.minimum(1, (25 - x) / 25)),
            'rendah': lambda x: np.maximum(0, np.minimum((x - 15) / 15, (45 - x) / 15)),
            'sedang': lambda x: np.maximum(0, np.minimum((x - 35) / 15, (65 - x) / 15)),
            'tinggi': lambda x: np.maximum(0, np.minimum((x - 55) / 15, (85 - x) / 15)),
            'sangat_tinggi': lambda x: np.maximum(0, np.minimum(1, (x - 75) / 10))
        }
    
    def fuzzification(self, inputs):
        fuzzified = {}
        
        for var_name, value in inputs.items():
            fuzzified[var_name] = {}
            for term, func in self.membership_functions[var_name].items():
                fuzzified[var_name][term] = func(value)
        
        return fuzzified
    
    def rule_evaluation(self, fuzzified_inputs):
        rules = [
            # Format: (kondisi, konsekuen, weight)
            (['ratio_upah_ump_rendah', 'ratio_upah_gk_sangat_rendah'], 'sangat_rendah', 1.0),
            (['ratio_upah_ump_rendah', 'ratio_upah_gk_rendah'], 'rendah', 1.0),
            (['ratio_upah_ump_sedang', 'ratio_upah_gk_sedang'], 'sedang', 1.0),
            (['ratio_upah_ump_tinggi', 'ratio_upah_gk_tinggi'], 'tinggi', 1.0),
            (['ratio_upah_ump_tinggi', 'ratio_upah_gk_sangat_tinggi'], 'sangat_tinggi', 1.0),
            (['efisiensi_pengeluaran_hemat', 'ratio_upah_gk_tinggi'], 'tinggi', 0.8),
            (['efisiensi_pengeluaran_boros', 'ratio_upah_ump_rendah'], 'rendah', 0.9),
        ]
        
        activated_rules = []
        
        for conditions, consequent, weight in rules:
            # Hitung activation strength menggunakan minimum
            activation_values = []
            
            for condition in conditions:
                parts = condition.split('_')
                var_name = '_'.join(parts[:-1])
                term = parts[-1]
                
                if var_name in fuzzified_inputs and term in fuzzified_inputs[var_name]:
                    activation_values.append(fuzzified_inputs[var_name][term])
            
            if activation_values:
                activation_strength = min(activation_values) * weight
                activated_rules.append((consequent, activation_strength))
        
        return activated_rules
    
    def defuzzification(self, activated_rules):
        if not activated_rules:
            return 50  # Default value
        
        # Agregasi menggunakan maximum
        aggregated = {}
        for consequent, strength in activated_rules:
            if consequent in aggregated:
                aggregated[consequent] = max(aggregated[consequent], strength)
            else:
                aggregated[consequent] = strength
        
        # Centroid calculation
        numerator = 0
        denominator = 0
        
        for consequent, strength in aggregated.items():
            if consequent == 'sangat_rendah':
                centroid = 12.5
            elif consequent == 'rendah':
                centroid = 30
            elif consequent == 'sedang':
                centroid = 50
            elif consequent == 'tinggi':
                centroid = 70
            elif consequent == 'sangat_tinggi':
                centroid = 87.5
            else:
                centroid = 50
            
            numerator += centroid * strength
            denominator += strength
        
        return numerator / denominator if denominator > 0 else 50
    
    def evaluate(self, ratio_ump, ratio_gk, efisiensi):
        inputs = {
            'ratio_upah_ump': ratio_ump,
            'ratio_upah_gk': ratio_gk,
            'efisiensi_pengeluaran': efisiensi
        }
        
        # Fuzzifikasi
        fuzzified = self.fuzzification(inputs)
        
        # Evaluasi aturan
        activated_rules = self.rule_evaluation(fuzzified)
        
        # Defuzzifikasi
        result = self.defuzzification(activated_rules)
        
        return result

class SugenoFuzzySystem:
    def __init__(self):
        self.name = "Sugeno"
        
    def create_membership_functions(self):
        # Sama dengan Mamdani untuk input
        self.membership_functions = {
            'ratio_upah_ump': {
                'rendah': lambda x: np.maximum(0, np.minimum(1, (1.0 - x) / 0.4)),
                'sedang': lambda x: np.maximum(0, np.minimum((x - 0.6) / 0.4, (1.5 - x) / 0.5)),
                'tinggi': lambda x: np.maximum(0, np.minimum(1, (x - 1.2) / 0.3))
            },
            'ratio_upah_gk': {
                'sangat_rendah': lambda x: np.maximum(0, np.minimum(1, (3 - x) / 3)),
                'rendah': lambda x: np.maximum(0, np.minimum((x - 2) / 1.5, (5 - x) / 1.5)),
                'sedang': lambda x: np.maximum(0, np.minimum((x - 4) / 2, (8 - x) / 2)),
                'tinggi': lambda x: np.maximum(0, np.minimum((x - 7) / 2, (12 - x) / 3)),
                'sangat_tinggi': lambda x: np.maximum(0, np.minimum(1, (x - 10) / 2))
            },
            'efisiensi_pengeluaran': {
                'boros': lambda x: np.maximum(0, np.minimum(1, (0.7 - x) / 0.2)),
                'normal': lambda x: np.maximum(0, np.minimum((x - 0.5) / 0.2, (0.9 - x) / 0.2)),
                'hemat': lambda x: np.maximum(0, np.minimum(1, (x - 0.8) / 0.1))
            }
        }
        
        # Output functions untuk Sugeno (linear functions)
        self.output_functions = {
            'sangat_rendah': lambda x1, x2, x3: 10 + 5*x1 + 3*x2 + 2*x3,
            'rendah': lambda x1, x2, x3: 25 + 8*x1 + 5*x2 + 3*x3,
            'sedang': lambda x1, x2, x3: 45 + 10*x1 + 8*x2 + 5*x3,
            'tinggi': lambda x1, x2, x3: 65 + 12*x1 + 10*x2 + 8*x3,
            'sangat_tinggi': lambda x1, x2, x3: 80 + 15*x1 + 12*x2 + 10*x3
        }
    
    def fuzzification(self, inputs):
        fuzzified = {}
        
        for var_name, value in inputs.items():
            fuzzified[var_name] = {}
            for term, func in self.membership_functions[var_name].items():
                fuzzified[var_name][term] = func(value)
        
        return fuzzified
    
    def rule_evaluation(self, fuzzified_inputs, original_inputs):
        rules = [
            (['ratio_upah_ump_rendah', 'ratio_upah_gk_sangat_rendah'], 'sangat_rendah', 1.0),
            (['ratio_upah_ump_rendah', 'ratio_upah_gk_rendah'], 'rendah', 1.0),
            (['ratio_upah_ump_sedang', 'ratio_upah_gk_sedang'], 'sedang', 1.0),
            (['ratio_upah_ump_tinggi', 'ratio_upah_gk_tinggi'], 'tinggi', 1.0),
            (['ratio_upah_ump_tinggi', 'ratio_upah_gk_sangat_tinggi'], 'sangat_tinggi', 1.0),
            (['efisiensi_pengeluaran_hemat', 'ratio_upah_gk_tinggi'], 'tinggi', 0.8),
            (['efisiensi_pengeluaran_boros', 'ratio_upah_ump_rendah'], 'rendah', 0.9),
        ]
        
        weighted_outputs = []
        total_weights = 0
        
        for conditions, consequent, weight in rules:
            # Hitung activation strength
            activation_values = []
            
            for condition in conditions:
                parts = condition.split('_')
                var_name = '_'.join(parts[:-1])
                term = parts[-1]
                
                if var_name in fuzzified_inputs and term in fuzzified_inputs[var_name]:
                    activation_values.append(fuzzified_inputs[var_name][term])
            
            if activation_values:
                activation_strength = min(activation_values) * weight
                
                # Hitung output menggunakan linear function
                output_value = self.output_functions[consequent](
                    original_inputs['ratio_upah_ump'],
                    original_inputs['ratio_upah_gk'],
                    original_inputs['efisiensi_pengeluaran']
                )
                
                weighted_outputs.append(activation_strength * output_value)
                total_weights += activation_strength
        
        return weighted_outputs, total_weights
    
    def defuzzification(self, weighted_outputs, total_weights):
        if total_weights == 0:
            return 50
        
        return sum(weighted_outputs) / total_weights
    
    def evaluate(self, ratio_ump, ratio_gk, efisiensi):
        inputs = {
            'ratio_upah_ump': ratio_ump,
            'ratio_upah_gk': ratio_gk,
            'efisiensi_pengeluaran': efisiensi
        }
        
        # Fuzzifikasi
        fuzzified = self.fuzzification(inputs)
        
        # Evaluasi aturan
        weighted_outputs, total_weights = self.rule_evaluation(fuzzified, inputs)
        
        # Defuzzifikasi
        result = self.defuzzification(weighted_outputs, total_weights)
        
        return result
    
class FuzzySystemEvaluator:
    def __init__(self):
        self.mamdani_system = MamdaniFuzzySystem()
        self.sugeno_system = SugenoFuzzySystem()
        
        # Inisialisasi membership functions
        self.mamdani_system.create_membership_functions()
        self.sugeno_system.create_membership_functions()
    
    def calculate_indicators(self, df):
        df = df.copy()
        df['upah_bulanan'] = df['upah_per_jam'] * 160
        df['ratio_upah_ump'] = df['upah_bulanan'] / df['ump']
        df['ratio_upah_gk'] = df['upah_bulanan'] / df['garis_kemiskinan']
        df['efisiensi_pengeluaran'] = df['garis_kemiskinan'] / df['pengeluaran_perkapita']
        
        return df
    
    def create_ground_truth(self, df):
        # Formula sederhana untuk ground truth
        normalized_ump = (df['ratio_upah_ump'] - df['ratio_upah_ump'].min()) / (df['ratio_upah_ump'].max() - df['ratio_upah_ump'].min())
        normalized_gk = (df['ratio_upah_gk'] - df['ratio_upah_gk'].min()) / (df['ratio_upah_gk'].max() - df['ratio_upah_gk'].min())
        normalized_eff = (df['efisiensi_pengeluaran'] - df['efisiensi_pengeluaran'].min()) / (df['efisiensi_pengeluaran'].max() - df['efisiensi_pengeluaran'].min())
        
        ground_truth = (normalized_ump * 0.4 + normalized_gk * 0.4 + normalized_eff * 0.2) * 100
        
        return ground_truth
    
    def evaluate_systems(self, df):
        print("\nTAHAP 2: IMPLEMENTASI SISTEM FUZZY")
        print("="*50)
        
        # Hitung indikator
        df = self.calculate_indicators(df)
        
        # Buat ground truth
        df['ground_truth'] = self.create_ground_truth(df)
        
        # Evaluasi Mamdani
        print("Menjalankan evaluasi Mamdani...")
        start_time = time.time()
        mamdani_results = []
        
        for _, row in df.iterrows():
            result = self.mamdani_system.evaluate(
                row['ratio_upah_ump'],
                row['ratio_upah_gk'],
                row['efisiensi_pengeluaran']
            )
            mamdani_results.append(result)
        
        mamdani_time = time.time() - start_time
        df['mamdani_score'] = mamdani_results
        
        # Evaluasi Sugeno
        print("Menjalankan evaluasi Sugeno...")
        start_time = time.time()
        sugeno_results = []
        
        for _, row in df.iterrows():
            result = self.sugeno_system.evaluate(
                row['ratio_upah_ump'],
                row['ratio_upah_gk'],
                row['efisiensi_pengeluaran']
            )
            sugeno_results.append(result)
        
        sugeno_time = time.time() - start_time
        df['sugeno_score'] = sugeno_results
        
        print(f"Mamdani selesai dalam {mamdani_time:.2f} detik")
        print(f"Sugeno selesai dalam {sugeno_time:.2f} detik")
        
        return df, mamdani_time, sugeno_time
    
    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Hitung akurasi berdasarkan threshold
        threshold = 10  # toleransi 10 poin
        accuracy = np.mean(np.abs(y_true - y_pred) <= threshold) * 100
        
        # Hitung F1-score berdasarkan kategori
        def categorize(scores):
            return pd.cut(scores, bins=[0, 25, 45, 65, 85, 100], 
                        labels=['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'])
        
        true_categories = categorize(y_true)
        pred_categories = categorize(y_pred)
        
        # Simplified F1-score calculation
        correct_predictions = (true_categories == pred_categories).sum()
        f1_score = correct_predictions / len(y_true) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Accuracy': accuracy,
            'F1-Score': f1_score
        }
    
    def compare_performance(self, df):
        print("\nTAHAP 3: EVALUASI PERFORMA SISTEM")
        print("="*50)
        
        # Hitung metrik untuk Mamdani
        mamdani_metrics = self.calculate_metrics(df['ground_truth'], df['mamdani_score'])
        
        # Hitung metrik untuk Sugeno
        sugeno_metrics = self.calculate_metrics(df['ground_truth'], df['sugeno_score'])
        
        # Tampilkan hasil perbandingan
        print("\nPERBANDINGAN PERFORMA:")
        print("-" * 60)
        print(f"{'Metrik':<15} {'Mamdani':<15} {'Sugeno':<15} {'Terbaik':<10}")
        print("-" * 60)
        
        for metric in mamdani_metrics.keys():
            mamdani_val = mamdani_metrics[metric]
            sugeno_val = sugeno_metrics[metric]
            
            # Tentukan mana yang lebih baik (untuk MSE, MAE, RMSE: lebih kecil lebih baik)
            if metric in ['MSE', 'MAE', 'RMSE']:
                better = 'Mamdani' if mamdani_val < sugeno_val else 'Sugeno'
            else:  # Untuk R², Accuracy, F1-Score: lebih besar lebih baik
                better = 'Mamdani' if mamdani_val > sugeno_val else 'Sugeno'
            
            print(f"{metric:<15} {mamdani_val:<15.3f} {sugeno_val:<15.3f} {better:<10}")
        
        return mamdani_metrics, sugeno_metrics
    
    def create_visualizations(self, df):
        print("\nMEMBUAT VISUALISASI...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Scatter plot Mamdani vs Ground Truth
        axes[0,0].scatter(df['ground_truth'], df['mamdani_score'], alpha=0.6, color='blue')
        axes[0,0].plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
        axes[0,0].set_xlabel('Ground Truth')
        axes[0,0].set_ylabel('Mamdani Prediction')
        axes[0,0].set_title('Mamdani vs Ground Truth')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot Sugeno vs Ground Truth
        axes[0,1].scatter(df['ground_truth'], df['sugeno_score'], alpha=0.6, color='green')
        axes[0,1].plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
        axes[0,1].set_xlabel('Ground Truth')
        axes[0,1].set_ylabel('Sugeno Prediction')
        axes[0,1].set_title('Sugeno vs Ground Truth')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Perbandingan distribusi hasil
        axes[0,2].hist(df['ground_truth'], alpha=0.5, label='Ground Truth', bins=20)
        axes[0,2].hist(df['mamdani_score'], alpha=0.5, label='Mamdani', bins=20)
        axes[0,2].hist(df['sugeno_score'], alpha=0.5, label='Sugeno', bins=20)
        axes[0,2].set_xlabel('Skor Kesejahteraan')
        axes[0,2].set_ylabel('Frekuensi')
        axes[0,2].set_title('Distribusi Skor')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Error distribution Mamdani
        mamdani_errors = df['mamdani_score'] - df['ground_truth']
        axes[1,0].hist(mamdani_errors, bins=20, alpha=0.7, color='blue')
        axes[1,0].axvline(mamdani_errors.mean(), color='red', linestyle='--', 
                        label=f'Mean Error: {mamdani_errors.mean():.2f}')
        axes[1,0].set_xlabel('Error (Predicted - Actual)')
        axes[1,0].set_ylabel('Frekuensi')
        axes[1,0].set_title('Distribusi Error Mamdani')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Error distribution Sugeno
        sugeno_errors = df['sugeno_score'] - df['ground_truth']
        axes[1,1].hist(sugeno_errors, bins=20, alpha=0.7, color='green')
        axes[1,1].axvline(sugeno_errors.mean(), color='red', linestyle='--',
                        label=f'Mean Error: {sugeno_errors.mean():.2f}')
        axes[1,1].set_xlabel('Error (Predicted - Actual)')
        axes[1,1].set_ylabel('Frekuensi')
        axes[1,1].set_title('Distribusi Error Sugeno')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Perbandingan langsung Mamdani vs Sugeno
        axes[1,2].scatter(df['mamdani_score'], df['sugeno_score'], alpha=0.6)
        axes[1,2].plot([0, 100], [0, 100], 'r--', label='Perfect Agreement')
        axes[1,2].set_xlabel('Mamdani Score')
        axes[1,2].set_ylabel('Sugeno Score')
        axes[1,2].set_title('Mamdani vs Sugeno')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('perbandingan_fuzzy_systems.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualisasi disimpan sebagai 'perbandingan_fuzzy_systems.png'")
    
    def generate_detailed_report(self, df, mamdani_metrics, sugeno_metrics, mamdani_time, sugeno_time, quality_report):
        print("\nTAHAP 4: ANALISIS DAN LAPORAN")
        print("="*50)
        
        # Buat laporan lengkap
        report = {
            'data_quality': quality_report,
            'dataset_info': {
                'total_samples': len(df),
                'provinces': len(df['provinsi'].unique()),
                'years': sorted(df['tahun'].unique()),
                'missing_values_after_cleaning': df.isnull().sum().sum()
            },
            'performance_comparison': {
                'mamdani': {
                    'metrics': mamdani_metrics,
                    'execution_time': mamdani_time
                },
                'sugeno': {
                    'metrics': sugeno_metrics,
                    'execution_time': sugeno_time
                }
            }
        }
        
        # Analisis per provinsi
        provincial_analysis = df.groupby('provinsi').agg({
            'ground_truth': 'mean',
            'mamdani_score': 'mean',
            'sugeno_score': 'mean',
            'ratio_upah_ump': 'mean',
            'ratio_upah_gk': 'mean',
            'efisiensi_pengeluaran': 'mean'
        }).round(2)
        
        # Ranking provinsi
        provincial_analysis['mamdani_rank'] = provincial_analysis['mamdani_score'].rank(ascending=False)
        provincial_analysis['sugeno_rank'] = provincial_analysis['sugeno_score'].rank(ascending=False)
        
        print("\nTOP 10 PROVINSI (Mamdani):")
        top_mamdani = provincial_analysis.nlargest(10, 'mamdani_score')
        for i, (prov, row) in enumerate(top_mamdani.iterrows(), 1):
            print(f"{i:2d}. {prov}: {row['mamdani_score']:.2f}")
        
        print("\nTOP 10 PROVINSI (Sugeno):")
        top_sugeno = provincial_analysis.nlargest(10, 'sugeno_score')
        for i, (prov, row) in enumerate(top_sugeno.iterrows(), 1):
            print(f"{i:2d}. {prov}: {row['sugeno_score']:.2f}")
        
        # Analisis kesesuaian ranking
        ranking_correlation = provincial_analysis[['mamdani_rank', 'sugeno_rank']].corr().iloc[0,1]
        print(f"\nKorelasi Ranking Mamdani-Sugeno: {ranking_correlation:.3f}")
        
        # Simpan hasil ke file
        self.save_detailed_results(df, provincial_analysis, report)
        
        return report
    
    def save_detailed_results(self, df, provincial_analysis, report):
        try:
            # Simpan hasil evaluasi lengkap
            df_export = df[[
                'tahun', 'provinsi', 'upah_per_jam', 'ump', 'garis_kemiskinan', 
                'pengeluaran_perkapita', 'ratio_upah_ump', 'ratio_upah_gk', 
                'efisiensi_pengeluaran', 'ground_truth', 'mamdani_score', 'sugeno_score'
            ]].round(2)
            
            df_export.to_csv('hasil_evaluasi_fuzzy_lengkap.csv', index=False)
            
            # Simpan analisis provinsi
            provincial_analysis.to_csv('analisis_per_provinsi.csv')
            
            # Simpan laporan dalam format teks
            with open('laporan_analisis_fuzzy.txt', 'w', encoding='utf-8') as f:
                f.write("LAPORAN ANALISIS SISTEM FUZZY KESEJAHTERAAN PEKERJA\n")
                f.write("="*60 + "\n\n")
                
                f.write("1. INFORMASI DATASET\n")
                f.write("-"*20 + "\n")
                f.write(f"Total sampel: {report['dataset_info']['total_samples']}\n")
                f.write(f"Jumlah provinsi: {report['dataset_info']['provinces']}\n")
                f.write(f"Tahun: {report['dataset_info']['years']}\n\n")
                
                f.write("2. KUALITAS DATA\n")
                f.write("-"*15 + "\n")
                for dataset, quality in report['data_quality'].items():
                    f.write(f"{dataset.upper()}:\n")
                    f.write(f"  - Total baris: {quality['total_rows']}\n")
                    f.write(f"  - Missing values: {quality['missing_values']} ({quality['missing_percentage']:.2f}%)\n")
                    f.write(f"  - Duplicate rows: {quality['duplicate_rows']}\n\n")
                
                f.write("3. PERFORMA SISTEM FUZZY\n")
                f.write("-"*25 + "\n")
                f.write("MAMDANI:\n")
                for metric, value in report['performance_comparison']['mamdani']['metrics'].items():
                    f.write(f"  - {metric}: {value:.3f}\n")
                f.write(f"  - Waktu eksekusi: {report['performance_comparison']['mamdani']['execution_time']:.2f} detik\n\n")
                
                f.write("SUGENO:\n")
                for metric, value in report['performance_comparison']['sugeno']['metrics'].items():
                    f.write(f"  - {metric}: {value:.3f}\n")
                f.write(f"  - Waktu eksekusi: {report['performance_comparison']['sugeno']['execution_time']:.2f} detik\n\n")
            
            print("\nFile tersimpan:")
            print("  hasil_evaluasi_fuzzy_lengkap.csv")
            print("  analisis_per_provinsi.csv")
            print("  laporan_analisis_fuzzy.txt")
            
        except Exception as e:
            print(f"Error menyimpan file: {e}")

def main():
    print("SISTEM FUZZY EVALUASI KESEJAHTERAAN PEKERJA")
    print("="*70)
    print("Implementasi dan Perbandingan Metode Mamdani vs Sugeno\n")
    
    # Tahap 1: Preprocessing Data
    preprocessor = DataPreprocessor()
    data, quality_report = preprocessor.load_and_validate_data()
    
    if data is None:
        print("Gagal memuat data. Program dihentikan.")
        return
    
    # Tahap 2: Evaluasi Sistem Fuzzy
    evaluator = FuzzySystemEvaluator()
    evaluated_data, mamdani_time, sugeno_time = evaluator.evaluate_systems(data)
    
    # Tahap 3: Perbandingan Performa
    mamdani_metrics, sugeno_metrics = evaluator.compare_performance(evaluated_data)
    
    # Tahap 4: Visualisasi
    evaluator.create_visualizations(evaluated_data)
    
    # Tahap 5: Laporan Detail
    report = evaluator.generate_detailed_report(
        evaluated_data, mamdani_metrics, sugeno_metrics, 
        mamdani_time, sugeno_time, quality_report
    )
    
    # Kesimpulan
    print("\nANALISIS SELESAI!")
    print("="*25)
    
    # Tentukan sistem terbaik
    mamdani_score = (mamdani_metrics['Accuracy'] + mamdani_metrics['F1-Score'] + 
                    (100 - mamdani_metrics['MAE'])) / 3
    sugeno_score = (sugeno_metrics['Accuracy'] + sugeno_metrics['F1-Score'] + 
                    (100 - sugeno_metrics['MAE'])) / 3
    
    best_system = "Mamdani" if mamdani_score > sugeno_score else "Sugeno"
    
    print(f"Sistem terbaik: {best_system}")
    print(f"Skor Mamdani: {mamdani_score:.2f}")
    print(f"Skor Sugeno: {sugeno_score:.2f}")
    print(f"Total sampel diproses: {len(evaluated_data)}")
    print(f"Provinsi dianalisis: {len(evaluated_data['provinsi'].unique())}")
    
    print(f"\nOutput Files:")
    print(f"  hasil_evaluasi_fuzzy_lengkap.csv")
    print(f"  analisis_per_provinsi.csv") 
    print(f"  laporan_analisis_fuzzy.txt")
    print(f"  perbandingan_fuzzy_systems.png")

if __name__ == "__main__":
    main()