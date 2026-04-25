import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix


class TraditionalSVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', test_size=0.2, random_state=42, scaler=None, verbose=True):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.scaler = scaler
        self.verbose = verbose
    def fit(self, X, y):
        """
        Train model
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        # scale
        if self.scaler is None:
            self.scaler = StandardScaler()
            
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # train SVM
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True
        )

        self.model.fit(X_train, y_train)

        # evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # precision, recall, f1 theo từng class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )

        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "confusion_matrix": cm
        }
        if self.verbose: 
            print(classification_report(y_test, y_pred))

        return metrics

    def predict(self, X):
        """
        Predict multiple samples
        """
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_one(self, features):
        """
        Predict single sample
        """
        features = np.array(features).reshape(1, -1)
        features = self.scaler.transform(features)
        return self.model.predict(features)[0]

    def get_model(self):
        """
        Trả model + scaler để utils save
        """
        return self.model, self.scaler

    def set_model(self, model, scaler):
        """
        Load model từ utils
        """
        self.model = model
        self.scaler = scaler