import cv2
import numpy as np
import os
from PIL import Image
import io

# TensorFlow import is commented out to avoid compatibility issues
# import tensorflow as tf

class FoodSpoilageDetector:
    """
    A class to detect signs of food spoilage in images.
    This model is designed to identify deteriorated or expired food items.
    """
    
    def __init__(self):
        """
        Initialize the food spoilage detection model.
        """
        self.model = self._load_model()
        self.detection_threshold = 0.6  # Higher threshold for spoilage claims
        self.input_size = (640, 480)    # Standard input size for the model
        
    def _load_model(self):
        """
        Load the pre-trained food spoilage detection model.
        
        Returns:
            A TensorFlow model instance or None if simulating
        """
        # In a production environment, we would load an actual TensorFlow model:
        # model_path = os.path.join(os.path.dirname(__file__), '../assets/models/spoilage_detection_model')
        # if os.path.exists(model_path):
        #     return tf.saved_model.load(model_path)
        
        # For this demo, we're simulating the model
        print("Food spoilage detection model not loaded - using simulated detection instead")
        return None
    
    def preprocess_image(self, image_bytes):
        """
        Preprocess the image for spoilage detection.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (TensorFlow models typically expect RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to the expected input dimensions
        image = cv2.resize(image, self.input_size)
        
        # Normalize pixel values to [0, 1]
        image = image / 255.0
        
        return image
    
    def detect_spoilage(self, image_bytes, detected_items=None):
        """
        Detect signs of food spoilage in an image.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            detected_items: Optional list of items detected in the image
            
        Returns:
            List of dictionaries with spoilage detection results
        """
        processed_image = self.preprocess_image(image_bytes)
        
        if self.model is not None:
            try:
                # For a real implementation with TensorFlow models:
                # spoilage_scores = self.model.predict(np.expand_dims(processed_image, 0))[0]
                # Process the spoilage scores...
                
                # For this demo, simulate detection
                return self._simulate_spoilage_detection(processed_image, detected_items)
                
            except Exception as e:
                print(f"Error during spoilage detection: {e}")
                return []
        else:
            # If model loading failed, use simulation
            return self._simulate_spoilage_detection(processed_image, detected_items)
    
    def _analyze_image_features(self, image):
        """
        Analyze image features to detect potential signs of spoilage.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary with image analysis results
        """
        # Image to BGR for analysis (OpenCV convention)
        img_bgr = (image * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        # Convert to different color spaces for analysis
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate features for spoilage detection
        features = {}
        
        # HSV color distribution - useful for detecting color changes in food
        features['hsv_means'] = [
            np.mean(img_hsv[:,:,0]),  # Hue
            np.mean(img_hsv[:,:,1]),  # Saturation
            np.mean(img_hsv[:,:,2])   # Value
        ]
        
        features['hsv_stds'] = [
            np.std(img_hsv[:,:,0]),  # Hue variability
            np.std(img_hsv[:,:,1]),  # Saturation variability
            np.std(img_hsv[:,:,2])   # Value variability
        ]
        
        # Texture analysis - helps detect mold, unusual textures
        # Edges, Laplacian for texture
        edges = cv2.Canny(img_gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        features['laplacian_variance'] = np.var(laplacian)
        
        # Look for discoloration patterns (potential mold, browning)
        # Green-blue hues can indicate mold
        mold_mask = cv2.inRange(img_hsv, (80, 50, 50), (130, 255, 255))
        features['mold_ratio'] = np.sum(mold_mask > 0) / (mold_mask.shape[0] * mold_mask.shape[1])
        
        # Brown hues can indicate decay
        brown_mask = cv2.inRange(img_hsv, (10, 50, 50), (30, 200, 150))
        features['browning_ratio'] = np.sum(brown_mask > 0) / (brown_mask.shape[0] * brown_mask.shape[1])
        
        return features
    
    def _simulate_spoilage_detection(self, image, detected_items=None):
        """
        Simulate spoilage detection with realistic features.
        Used when a real model is not available.
        
        Args:
            image: Preprocessed image
            detected_items: List of items detected in the image
            
        Returns:
            List of dictionaries with simulated spoilage detection results
        """
        # Use image analysis to make more realistic predictions
        features = self._analyze_image_features(image)
        
        # Items prone to visible spoilage
        highly_perishable = ['Lettuce', 'Tomatoes', 'Strawberries', 'Herbs', 'Fish', 
                           'Chicken', 'Milk', 'Yogurt', 'Bananas', 'Berries']
        
        medium_perishable = ['Bell Peppers', 'Apples', 'Oranges', 'Cheese', 'Tofu',
                           'Bread', 'Hummus']
        
        # If no detected items provided, use empty list
        detected_items = detected_items or []
                
        # Potential spoilage issues
        spoilage_issues = {
            'mold': {'items': ['Bread', 'Cheese', 'Fruit', 'Tomatoes', 'Berries'], 
                     'trigger': features['mold_ratio'] > 0.02},
            'wilting': {'items': ['Lettuce', 'Herbs', 'Spinach', 'Basil'], 
                       'trigger': features['hsv_means'][1] < 60 and features['edge_density'] > 0.2},
            'browning': {'items': ['Apples', 'Bananas', 'Avocados', 'Lettuce'], 
                        'trigger': features['browning_ratio'] > 0.1},
            'texture_change': {'items': ['Meat', 'Fish', 'Chicken', 'Tofu'], 
                             'trigger': features['laplacian_variance'] > 500},
            'discoloration': {'items': ['Meat', 'Fish', 'Bell Peppers', 'Broccoli'], 
                            'trigger': features['hsv_stds'][0] > 50}
        }
        
        spoiled_items = []
        
        # First check detected items
        for item_data in detected_items:
            item_name = item_data['item']
            base_prob = 0.0
            
            # Higher base probability for highly perishable items
            if any(perishable in item_name for perishable in highly_perishable):
                base_prob = 0.3
            elif any(perishable in item_name for perishable in medium_perishable):
                base_prob = 0.15
                
            # Check each spoilage issue
            for issue, data in spoilage_issues.items():
                if any(keyword in item_name for keyword in data['items']):
                    if data['trigger']:
                        confidence = base_prob + np.random.uniform(0.3, 0.5)
                        # Cap at 0.95 and ensure at least 0.6 for a positive detection
                        confidence = min(0.95, max(0.6, confidence))
                        
                        spoiled_items.append({
                            'item': item_name,
                            'confidence': confidence,
                            'issue': issue.replace('_', ' ').title()
                        })
                        break
        
        # If no spoilage detected but image features strongly suggest spoilage,
        # add a generic detection
        if not spoiled_items and (features['mold_ratio'] > 0.03 or 
                                 features['browning_ratio'] > 0.15 or
                                 features['laplacian_variance'] > 700):
            # Make an educated guess about what might be spoiled
            if features['mold_ratio'] > 0.03:
                item = "Possible Moldy Item"
                issue = "Mold Growth"
            elif features['browning_ratio'] > 0.15:
                item = "Possible Browning Food"
                issue = "Discoloration"
            else:
                item = "Possible Spoiled Item"
                issue = "Texture Changes"
                
            spoiled_items.append({
                'item': item,
                'confidence': min(0.95, max(0.6, np.random.uniform(0.65, 0.85))),
                'issue': issue
            })
            
        # Rarely report spoilage (we don't want false positives)
        # So if detected items seem low risk, often return empty
        if (not any(data['trigger'] for data in spoilage_issues.values()) and 
            np.random.random() > 0.2):
            return []
        
        # Limit to at most 2 detections to keep results reasonable
        return spoiled_items[:2]

def get_spoilage_detector():
    """
    Factory function to get the food spoilage detector instance.
    
    Returns:
        An instance of FoodSpoilageDetector
    """
    return FoodSpoilageDetector()
