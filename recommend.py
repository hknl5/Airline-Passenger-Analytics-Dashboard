def recommend_products(age, travel_purpose, ticket_class):
    """
    Rule-based product recommendation system
    
    Args:
        age (int): Passenger age
        travel_purpose (str): Purpose of travel
        ticket_class (str): Ticket class
    
    Returns:
        list: Recommended products
    """
    recommendations = []
    
    # Age-based recommendations
    if age < 30:
        recommendations.extend(['Snacks', 'Makeup'])
    elif age >= 30 and age < 50:
        recommendations.extend(['Electronics', 'Perfume'])
    else:
        recommendations.extend(['Books', 'Health Products'])
    
    # Travel purpose-based recommendations
    if travel_purpose.lower() in ['business', 'work']:
        recommendations.extend(['Electronics', 'Luxury Bags'])
    elif travel_purpose.lower() in ['leisure', 'vacation', 'holiday']:
        recommendations.extend(['Snacks', 'Souvenirs'])
    elif travel_purpose.lower() in ['medical', 'health']:
        recommendations.extend(['Health Products', 'Books'])
    
    # Ticket class-based recommendations
    if ticket_class.lower() == 'first':
        recommendations.extend(['Watches', 'Perfume', 'Luxury Bags'])
    elif ticket_class.lower() == 'business':
        recommendations.extend(['Electronics', 'Watches'])
    else:  # Economy
        recommendations.extend(['Snacks', 'Books'])
    
    # Remove duplicates and return top 5
    unique_recommendations = list(set(recommendations))
    return unique_recommendations[:5]

def get_all_product_categories():
    """Return all available product categories"""
    return [
        'Electronics', 'Snacks', 'Makeup', 'Perfume', 
        'Watches', 'Luxury Bags', 'Books', 'Health Products', 
        'Souvenirs', 'Clothing', 'Jewelry', 'Alcohol'
    ]

# Test the recommendation function
if __name__ == "__main__":
    # Test cases
    test_cases = [
        (25, 'Leisure', 'Economy'),
        (35, 'Business', 'First'),
        (45, 'Medical', 'Business'),
        (28, 'Vacation', 'Economy')
    ]
    
    for age, purpose, ticket_class in test_cases:
        recommendations = recommend_products(age, purpose, ticket_class)
        print(f"Age: {age}, Purpose: {purpose}, Class: {ticket_class}")
        print(f"Recommendations: {recommendations}")
        print()
