import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import copy

class TabularGRPOClassifier:
    
    def __init__(self, model_class, input_dim, num_classes, group_size=10, clip_epsilon=0.2, kl_coef=0.01, lr=1e-4, weight_decay=1e-5, epochs=200, batch_size=64, precision_threshold=0.95):
        """
        Initialize the TabularGRPOClassifier object.

        Parameters
        ----------
        model_class : type
            The class of the model to be trained.
        input_dim : int
            The input dimensionality of the model.
        num_classes : int
            The number of output classes of the model.
        group_size : int, optional
            The size of the groups used in the GRPO algorithm. Defaults to 10.
        clip_epsilon : float, optional
            The clipping value used for the policy ratio. Defaults to 0.2.
        kl_coef : float, optional
            The coefficient of the KL divergence term in the loss. Defaults to 0.01.
        lr : float, optional
            The learning rate of the model. Defaults to 1e-4.
        weight_decay : float, optional
            The weight decay coefficient of the model. Defaults to 1e-5.
        epochs : int, optional
            The number of training epochs. Defaults to 200.
        batch_size : int, optional
            The batch size of the training data. Defaults to 64.
        precision_threshold : float, optional
            The precision threshold used to determine when to stop training. Defaults to 0.95.
        """
        self.model_class = model_class
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.precision_threshold = precision_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
    
    def _sample_actions(self, logits):
        """Sample a batch of actions from a given batch of logits.

        Parameters
        ----------
        logits : torch.Tensor
            The batch of logits to sample from.

        Returns
        -------
        actions : torch.Tensor
            The sampled actions.
        probs : torch.Tensor
            The probabilities from which the actions were sampled.
        """
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, self.group_size, replacement=True)
        return actions, probs

    def _compute_rewards(self, actions, labels):
        """
        Compute rewards from sampled actions and their corresponding labels.

        Parameters
        ----------
        actions : torch.Tensor
            A tensor of shape (batch_size, num_samples) containing sampled actions.
        labels : torch.Tensor
            A tensor of shape (batch_size) containing the corresponding labels.

        Returns
        -------
        rewards : torch.Tensor
            A tensor of shape (batch_size, num_samples) containing the binary rewards.
        """
        labels_exp = labels.unsqueeze(1).expand_as(actions)
        return (actions == labels_exp).float()

    def _compute_group_advantages(self, rewards, std_eps=1e-8):
        """
        Computes group relative advantages from rewards.

        Given a tensor of rewards, computes the group relative advantages by subtracting the
        group mean and dividing by the group standard deviation. A small epsilon value is
        added to the standard deviation to prevent division by zero.

        Args:
            rewards (torch.Tensor): A tensor of shape (batch_size, num_samples) containing rewards.
            std_eps (float, optional): A small value added to the standard deviation to prevent division by zero.
                Defaults to 1e-8.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_samples) containing the group relative advantages.
        """
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True) + std_eps
        return (rewards - mean_r) / std_r

    def _compute_kl(self, old_probs, current_probs):
        """
        Computes the KL divergence between two categorical distributions for each sample.

        Parameters
        ----------
        old_probs : torch.Tensor
            A tensor of shape (batch_size, num_classes) containing the old policy probabilities.
        current_probs : torch.Tensor
            A tensor of shape (batch_size, num_classes) containing the current policy probabilities.

        Returns
        -------
        mean_kl : float
            The mean KL divergence between the old and current policy distributions.
        """
        kl = (old_probs * (old_probs.log() - current_probs.log())).sum(dim=1)
        return kl.mean()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model using the TabularGRPO algorithm.

        This method trains the model using the Generalized Reward Policy Optimization (GRPO) algorithm
        over the specified number of epochs. Training can be early stopped if the precision on the 
        validation set exceeds the specified threshold.

        Parameters
        ----------
        X_train : array-like
            The training input samples.
        y_train : array-like
            The target values for the training input samples.
        X_val : array-like, optional
            The validation input samples. Default is None.
        y_val : array-like, optional
            The target values for the validation input samples. Default is None.

        Notes
        -----
        If both X_val and y_val are provided, the method evaluates the model's precision on the
        validation set at the end of each epoch. If the precision exceeds the precision_threshold,
        training is halted early.

        Examples
        --------
        from tabulargrpo_classifier import TabularGRPOClassifier
        from models.transformer_moe import MoETransformer
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=5000, n_features=14, n_classes=2, random_state=42)
 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        classifier = TabularGRPOClassifier(model_class=MoETransformer,input_dim=14, num_classes=2, epochs=10,group_size=10)
        classifier.fit(X_train,y_train)

        """

        X_train = self.scaler.fit_transform(X_train)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = self.model_class(input_dim=self.input_dim, num_classes=self.num_classes).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.model.train()
        for epoch in range(self.epochs):
            old_model = copy.deepcopy(self.model)
            old_model.eval()
            total_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                logits_current = self.model(data)
                with torch.no_grad():
                    logits_old = old_model(data)
                
                actions, old_probs_full = self._sample_actions(logits_old)
                current_probs_full = F.softmax(logits_current, dim=-1)
                p_old = torch.gather(old_probs_full, 1, actions)
                p_current = torch.gather(current_probs_full, 1, actions)
                
                rewards = self._compute_rewards(actions, target)
                advantages = self._compute_group_advantages(rewards)
                
                ratio = p_current / (p_old + 1e-8)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                surrogate_loss = torch.where(advantages >= 0, torch.min(unclipped, clipped), torch.max(unclipped, clipped))
                loss_policy = -surrogate_loss.mean()
                kl_loss = self._compute_kl(old_probs_full, current_probs_full)
                loss = loss_policy + self.kl_coef * kl_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{self.epochs} | GRPO Loss: {avg_loss:.4f}')
            
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, verbose=False)
                if val_metrics['precision'] >= self.precision_threshold:
                    print(f"Early stopping at epoch {epoch+1} due to high precision: {val_metrics['precision']:.4f}")
                    break
    
    def predict(self, X_test):
        """
        Make predictions on a test set.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input data to make predictions on.

        Returns
        -------
        preds : array-like of shape (n_samples,)
            The predicted labels.
        """
        X_test = self.scaler.transform(X_test)
        test_tensor = torch.FloatTensor(X_test).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(test_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate the model on a test dataset and compute performance metrics.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input data to evaluate the model on.
        y_test : array-like of shape (n_samples,)
            The true labels for X_test.
        verbose : bool, optional
            If True, prints evaluation metrics. Default is True.

        Returns
        -------
        dict
            A dictionary containing the following evaluation metrics:
            - 'accuracy': The accuracy score of the model predictions.
            - 'precision': The precision score of the model predictions.
            - 'f1': The F1 score of the model predictions.
            - 'auc': The ROC AUC score of the model predictions.
        """

        X_test = self.scaler.transform(X_test)
        test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test = np.array(y_test)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(test_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
        if self.num_classes > 2:
            auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
            precision = precision_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')
        else:
            auc = roc_auc_score(y_test, probs[:, 1])
            precision = precision_score(y_test, preds)
            f1 = f1_score(y_test, preds)
        
        acc = accuracy_score(y_test, preds)
        
        if verbose:
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {auc:.4f}")
        
        return {'accuracy': acc, 'precision': precision, 'f1': f1, 'auc': auc}
