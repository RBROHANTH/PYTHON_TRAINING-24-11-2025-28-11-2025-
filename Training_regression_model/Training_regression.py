import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class RegressionModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate sample data for regression"""
        print("Generating sample data...")
        
        # Generate features
        X = np.random.randn(n_samples, 4)
        
        # Create target variable with some relationship to features
        y = (2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 
             0.5 * X[:, 3] + np.random.normal(0, 0.5, n_samples))
        
        # Create DataFrame for better handling
        feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']
        self.df = pd.DataFrame(X, columns=feature_names)
        self.df['Target'] = y
        
        print(f"Dataset created with {n_samples} samples and {len(feature_names)} features")
        return self.df
    
    def load_csv_data(self, filepath):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"Data loaded from {filepath}")
            print(f"Shape: {self.df.shape}")
            print("\nFirst few rows:")
            print(self.df.head())
            
            # Preprocess categorical variables if it's gemstone data
            if 'cut' in self.df.columns and 'color' in self.df.columns:
                self.df = self.preprocess_gemstone_data()
            
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_gemstone_data(self):
        """Preprocess gemstone dataset - handle categorical variables"""
        print("Preprocessing gemstone data...")
        
        # Drop id column if exists
        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)
        
        # Clean numeric columns - handle invalid values
        numeric_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
        for col in numeric_cols:
            if col in self.df.columns:
                # Convert to numeric, replacing invalid values with NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Drop rows with NaN values
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        final_rows = len(self.df)
        print(f"Removed {initial_rows - final_rows} rows with invalid data")
        
        # Remove outliers (0 values in x, y, z which are physical dimensions)
        self.df = self.df[(self.df['x'] > 0) & (self.df['y'] > 0) & (self.df['z'] > 0)]
        print(f"Removed outliers, final dataset: {len(self.df)} rows")
        
        # Encode categorical variables
        categorical_cols = ['cut', 'color', 'clarity']
        
        for col in categorical_cols:
            if col in self.df.columns:
                # Convert to categorical codes
                self.df[col] = pd.Categorical(self.df[col]).codes
                print(f"Encoded {col} as numerical values")
        
        print("Gemstone data preprocessing completed")
        return self.df
    
    def explore_data(self):
        """Explore and visualize the data"""
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        # Correlation matrix
        plt.figure(figsize=(12, 8))
        
        # Correlation heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        # Distribution plots
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Target distribution
            plt.subplot(2, 2, 2)
            if 'Target' in self.df.columns:
                plt.hist(self.df['Target'], bins=30, alpha=0.7, color='skyblue')
                plt.title('Target Variable Distribution')
                plt.xlabel('Target')
                plt.ylabel('Frequency')
            
            # Feature distributions
            plt.subplot(2, 2, 3)
            for i, col in enumerate(numeric_cols[:4]):
                if col != 'Target':
                    plt.plot(self.df[col], alpha=0.7, label=col)
            plt.title('Feature Distributions')
            plt.legend()
            
            # Scatter plot (if target exists)
            plt.subplot(2, 2, 4)
            if 'Target' in self.df.columns and len(numeric_cols) > 1:
                feature_col = [col for col in numeric_cols if col != 'Target'][0]
                plt.scatter(self.df[feature_col], self.df['Target'], alpha=0.6)
                plt.xlabel(feature_col)
                plt.ylabel('Target')
                plt.title(f'{feature_col} vs Target')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_data(self, target_column='Target', test_size=0.2):
        """Prepare data for training"""
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        print(f"\nPreparing data with target column: {target_column}")
        
        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
    
    def train_models(self):
        """Train multiple regression models"""
        if self.X_train is None:
            print("Data not prepared. Run prepare_data() first.")
            return
        
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Initialize models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if 'Forest' in name:
                model.fit(self.X_train, self.y_train)
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
            else:
                model.fit(self.X_train_scaled, self.y_train)
                train_pred = model.predict(self.X_train_scaled)
                test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            train_mse = mean_squared_error(self.y_train, train_pred)
            test_mse = mean_squared_error(self.y_test, test_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            
            # Store model and metrics
            self.models[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'predictions': test_pred
            }
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MSE: {test_mse:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all trained models"""
        if not self.models:
            print("No models trained. Run train_models() first.")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Create comparison DataFrame
        results = []
        for name, metrics in self.models.items():
            results.append({
                'Model': name,
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Test MSE': metrics['test_mse'],
                'Test MAE': metrics['test_mae']
            })
        
        results_df = pd.DataFrame(results)
        print("\nModel Comparison:")
        print(results_df.round(4))
        
        # Find best model
        best_model = results_df.loc[results_df['Test R²'].idxmax()]
        print(f"\nBest Model: {best_model['Model']} (Test R²: {best_model['Test R²']:.4f})")
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² comparison
        axes[0, 0].bar(results_df['Model'], results_df['Test R²'], color='skyblue')
        axes[0, 0].set_title('Test R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MSE comparison
        axes[0, 1].bar(results_df['Model'], results_df['Test MSE'], color='lightcoral')
        axes[0, 1].set_title('Test MSE Comparison')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Predictions vs Actual for best model
        best_model_name = best_model['Model']
        best_predictions = self.models[best_model_name]['predictions']
        
        axes[1, 0].scatter(self.y_test, best_predictions, alpha=0.6)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title(f'Predictions vs Actual ({best_model_name})')
        
        # Residuals plot
        residuals = self.y_test - best_predictions
        axes[1, 1].scatter(best_predictions, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title(f'Residuals Plot ({best_model_name})')
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def make_prediction(self, new_data):
        """Make predictions on new data"""
        if not self.models:
            print("No models trained. Run train_models() first.")
            return
        
        # Get best model
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x]['test_r2'])
        best_model = self.models[best_model_name]['model']
        
        print(f"Making predictions using {best_model_name}")
        
        # Prepare data
        if isinstance(new_data, list):
            new_data = np.array(new_data).reshape(1, -1)
        
        # Scale if needed
        if 'Forest' not in best_model_name:
            new_data = self.scaler.transform(new_data)
        
        prediction = best_model.predict(new_data)
        print(f"Prediction: {prediction[0]:.4f}")
        
        return prediction[0]

def main():
    """Main function to run the regression pipeline"""
    print("="*60)
    print("REGRESSION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize model
    reg_model = RegressionModel()
    
    # Option 1: Load gemstone.csv data
    print("\n1. Loading gemstone data from CSV...")
    data = reg_model.load_csv_data('gemstone.csv')
    if data is None:
        print("Failed to load gemstone.csv, using sample data instead...")
        data = reg_model.generate_sample_data(n_samples=1000)
        target_col = 'Target'
    else:
        target_col = 'price'  # gemstone dataset uses 'price' as target

    # Explore data
    print("\n2. Exploring data...")
    reg_model.explore_data()
    
    # Prepare data
    print("\n3. Preparing data...")
    reg_model.prepare_data(target_column=target_col)
    
    # Train models
    print("\n4. Training models...")
    reg_model.train_models()
    
    # Evaluate models
    print("\n5. Evaluating models...")
    results = reg_model.evaluate_models()
    
    # Make sample prediction
    print("\n6. Making sample prediction...")
    if target_col == 'price':  # Gemstone dataset
        # Sample: [carat, cut, color, clarity, depth, table, x, y, z]
        sample_input = [1.0, 2, 3, 1, 61.5, 57.0, 6.5, 6.5, 4.0]  # Example gemstone
        print("Sample gemstone: 1.0 carat, cut=2, color=3, clarity=1, depth=61.5, table=57.0, x=6.5, y=6.5, z=4.0")
    else:  # Synthetic dataset
        sample_input = [0.5, -0.3, 0.8, -0.2]  # Example input
    prediction = reg_model.make_prediction(sample_input)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
