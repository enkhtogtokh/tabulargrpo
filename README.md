# TabularGRPO

## Overview
TabularGRPO 

### Dataset
Dataset download url: https://zenodo.org/records/15081413.
After you download, please drop in data/ folder

### Project Structure
The project structure is as follows:

- `tabulargrpo`: Root directory
	- `README.md`: the reamde
	- `data`: Data directory
		- `dataset.py`: Training Testing data preperations
		- `synthetic_data_small.csv`: Download dataset and drop here
	- `models`: Model directory
		- `MoETransformer.py`: MoETransformer model	 
	- `tabulargrpo_classifier.py`: Training script
	- `demo.py`: Demo script

## Features
- **Tabular Classifier**: Tabular Data Training 
- **MoE Transformer**: MoE Transformer
- **GRPO**: GRPO
 

## Installation
To install the TabularGRPO project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tabulargrpo.git
    ```
2. Navigate to the project directory:
    ```bash
    cd tabulargrpo
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To use the TabularGRPO train, run the following command:
```bash
        from tabulargrpo_classifier import TabularGRPOClassifier
        from models.transformer_moe import MoETransformer
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=5000, n_features=14, n_classes=2, random_state=42)
 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        classifier = TabularGRPOClassifier(model_class=MoETransformer,input_dim=14, num_classes=2, epochs=10,group_size=10)
        classifier.fit(X_train,y_train)
```

To use the TabularGRPO evaluate, run the following command:
```bash
        from tabulargrpo_classifier import TabularGRPOClassifier
        from models.transformer_moe import MoETransformer
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=5000, n_features=14, n_classes=2, random_state=42)
 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        classifier = TabularGRPOClassifier(model_class=MoETransformer,input_dim=14, num_classes=2, epochs=10,group_size=10)
        classifier.fit(X_train,y_train)
        classifier.evaluate(X_test, y_test)
```

To use the TabularGRPO predict, run the following command:
```bash
        from tabulargrpo_classifier import TabularGRPOClassifier
        from models.transformer_moe import MoETransformer
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=5000, n_features=14, n_classes=2, random_state=42)
 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        classifier = TabularGRPOClassifier(model_class=MoETransformer,input_dim=14, num_classes=2, epochs=10,group_size=10)
        classifier.fit(X_train,y_train)
        classifier.evaluate(X_test, y_test)
        data = [[0,0,0,0]]
        p = classifier.predict(data)
        print(p)
```

To use the TabularGRPO demo, run the following command:
```bash
        python demo.py
```
 

## Contributing
We welcome contributions to the TabularGRPO project. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your branch.
4. Create a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or inquiries, please contact us at [enkhtogtokh.java@gmail.com](mailto:enkhtogtokh.java@gmail.com).
