import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def create_feature_importance_plot(feature_importance_df):
    """Create feature importance bar plot"""
    feature_importance_df = pd.DataFrame(feature_importance_df)
    return px.bar(
        feature_importance_df.head(10),
        x='Feature',
        y='Importance',
        title='Top Feature Importance'
    )

def create_confusion_matrix_plot(conf_matrix):
    """Create confusion matrix heatmap"""
    return px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual"),
        x=[f"Class {i}" for i in range(conf_matrix.shape[1])],
        y=[f"Class {i}" for i in range(conf_matrix.shape[0])],
        title="Confusion Matrix",
        color_continuous_scale='RdBu'
    )

def create_roc_curve_plot(y_test, y_pred):
    """Create ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig = px.line(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.2f})',
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    return fig

def create_decision_boundary_plot(X, y, model, scaler):
    """Create 2D decision boundary plot for SVM (only for 2D data or using PCA)"""
    # If data has more than 2 dimensions, use PCA
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X
    
    # Create mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Get predictions for mesh grid points
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # Create plot
    fig = go.Figure()
    
    # Add decision boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.02),
        y=np.arange(y_min, y_max, 0.02),
        z=Z,
        showscale=False,
        colorscale='RdBu'
    ))
    
    # Add scatter plot of actual points
    for i in np.unique(y):
        mask = y == i
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode='markers',
            name=f'Class {i}'
        ))
    
    fig.update_layout(
        title='SVM Decision Boundary',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2'
    )
    
    return fig

def create_kernel_comparison_plot(X_test, y_test, y_pred):
    """Create confusion matrix with kernel performance comparison"""
    # Calculate metrics for actual vs predicted
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        marginal_x='histogram',
        marginal_y='histogram',
        title='Actual vs Predicted Values',
        labels={'x': 'Actual Values', 'y': 'Predicted Values'}
    )
    
    return fig

def create_support_vectors_plot(model, X):
    """Create visualization of support vectors"""
    if hasattr(model, 'support_vectors_'):
        support_vectors = model.support_vectors_
        
        # Use PCA if more than 2 dimensions
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            sv_2d = pca.transform(support_vectors)
        else:
            X_2d = X
            sv_2d = support_vectors
        
        fig = go.Figure()
        
        # Plot all points
        fig.add_trace(go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode='markers',
            name='Data Points',
            marker=dict(size=8, color='blue', opacity=0.5)
        ))
        
        # Plot support vectors
        fig.add_trace(go.Scatter(
            x=sv_2d[:, 0],
            y=sv_2d[:, 1],
            mode='markers',
            name='Support Vectors',
            marker=dict(size=12, color='red', symbol='circle-open')
        ))
        
        fig.update_layout(
            title='Support Vectors Visualization',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2'
        )
        
        return fig