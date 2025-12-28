"""
E-commerce Clickstream Data Generator

Generates realistic user clickstream data for testing data pipeline.
Simulates user sessions with various event types (page views, clicks, add to cart, purchase).
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import csv


class ClickstreamGenerator:
    """Generate realistic e-commerce clickstream data"""

    # Event types with their probabilities
    EVENT_TYPES = {
        'page_view': 0.5,
        'product_click': 0.25,
        'add_to_cart': 0.15,
        'remove_from_cart': 0.03,
        'checkout': 0.05,
        'purchase': 0.02
    }

    # Product categories
    CATEGORIES = [
        'Electronics', 'Clothing', 'Books', 'Home & Garden',
        'Sports & Outdoors', 'Beauty', 'Toys', 'Food & Beverage'
    ]

    # Page types
    PAGES = [
        'home', 'category', 'product', 'cart',
        'checkout', 'account', 'search'
    ]

    # Device types
    DEVICES = ['desktop', 'mobile', 'tablet']

    # Browsers
    BROWSERS = ['Chrome', 'Firefox', 'Safari', 'Edge']

    def __init__(self, num_users: int = 1000, num_products: int = 500):
        """
        Initialize the generator

        Args:
            num_users: Number of unique users to simulate
            num_products: Number of unique products in catalog
        """
        self.num_users = num_users
        self.num_products = num_products
        self.user_ids = [f"user_{i:06d}" for i in range(num_users)]
        self.product_ids = [f"prod_{i:05d}" for i in range(num_products)]

    def generate_session(self, user_id: str, session_start: datetime) -> List[Dict]:
        """
        Generate a single user session with multiple events

        Args:
            user_id: Unique user identifier
            session_start: Session start timestamp

        Returns:
            List of event dictionaries
        """
        session_id = str(uuid.uuid4())
        num_events = random.randint(3, 20)
        events = []

        current_time = session_start
        device = random.choice(self.DEVICES)
        browser = random.choice(self.BROWSERS)

        # Session cart state
        cart_items = set()

        for _ in range(num_events):
            # Select event type based on probabilities
            event_type = random.choices(
                list(self.EVENT_TYPES.keys()),
                weights=list(self.EVENT_TYPES.values())
            )[0]

            # Generate event
            event = {
                'event_id': str(uuid.uuid4()),
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': current_time.isoformat(),
                'event_type': event_type,
                'page_type': random.choice(self.PAGES),
                'device': device,
                'browser': browser,
                'ip_address': self._generate_ip(),
                'referrer': self._generate_referrer()
            }

            # Add product-specific fields for relevant events
            if event_type in ['product_click', 'add_to_cart', 'remove_from_cart']:
                product_id = random.choice(self.product_ids)
                event['product_id'] = product_id
                event['category'] = random.choice(self.CATEGORIES)
                event['price'] = round(random.uniform(9.99, 999.99), 2)

                if event_type == 'add_to_cart':
                    cart_items.add(product_id)
                    event['cart_size'] = len(cart_items)
                elif event_type == 'remove_from_cart' and cart_items:
                    cart_items.discard(product_id)
                    event['cart_size'] = len(cart_items)

            # Purchase event includes cart value
            if event_type == 'purchase' and cart_items:
                event['cart_value'] = round(random.uniform(50, 2000), 2)
                event['num_items'] = len(cart_items)
                event['payment_method'] = random.choice(['credit_card', 'debit_card', 'paypal', 'apple_pay'])

            # Search events
            if event['page_type'] == 'search':
                event['search_query'] = random.choice([
                    'laptop', 'headphones', 'shoes', 'book', 'camera',
                    'phone', 'watch', 'desk', 'chair', 'backpack'
                ])

            events.append(event)

            # Increment time (1-300 seconds between events)
            current_time += timedelta(seconds=random.randint(1, 300))

        return events

    def _generate_ip(self) -> str:
        """Generate a random IP address"""
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

    def _generate_referrer(self) -> str:
        """Generate a referrer URL"""
        referrers = [
            'direct', 'google.com', 'facebook.com', 'instagram.com',
            'twitter.com', 'email_campaign', 'organic_search', 'paid_ad'
        ]
        return random.choice(referrers)

    def generate_clickstream_data(
        self,
        num_sessions: int = 10000,
        start_date: datetime = None,
        days: int = 30
    ) -> List[Dict]:
        """
        Generate complete clickstream dataset

        Args:
            num_sessions: Number of sessions to generate
            start_date: Start date for data generation
            days: Number of days to spread data across

        Returns:
            List of all events
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)

        all_events = []

        print(f"Generating {num_sessions} sessions...")
        for i in range(num_sessions):
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_sessions} sessions...")

            # Random user and session start time
            user_id = random.choice(self.user_ids)
            session_start = start_date + timedelta(
                seconds=random.randint(0, days * 24 * 60 * 60)
            )

            session_events = self.generate_session(user_id, session_start)
            all_events.extend(session_events)

        print(f"Generated {len(all_events)} total events")
        return all_events

    def save_to_csv(self, events: List[Dict], output_path: str):
        """Save events to CSV file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if not events:
            print("No events to save")
            return

        # Get all unique keys from all events
        all_keys = set()
        for event in events:
            all_keys.update(event.keys())

        fieldnames = sorted(all_keys)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

        print(f"Saved {len(events)} events to {output_path}")

    def save_to_json(self, events: List[Dict], output_path: str):
        """Save events to JSON file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2)

        print(f"Saved {len(events)} events to {output_path}")

    def save_to_jsonl(self, events: List[Dict], output_path: str):
        """Save events to JSON Lines file (one JSON per line)"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')

        print(f"Saved {len(events)} events to {output_path}")


def main():
    """Main execution function"""
    # Configuration
    NUM_USERS = 1000
    NUM_PRODUCTS = 500
    NUM_SESSIONS = 10000
    DAYS = 30

    # Initialize generator
    generator = ClickstreamGenerator(
        num_users=NUM_USERS,
        num_products=NUM_PRODUCTS
    )

    # Generate data
    print("Starting clickstream data generation...")
    events = generator.generate_clickstream_data(
        num_sessions=NUM_SESSIONS,
        days=DAYS
    )

    # Save in multiple formats
    base_path = Path(__file__).parent.parent.parent / 'data' / 'raw'

    generator.save_to_csv(events, str(base_path / 'clickstream_events.csv'))
    generator.save_to_jsonl(events, str(base_path / 'clickstream_events.jsonl'))

    # Generate product catalog as well
    print("\nGenerating product catalog...")
    products = []
    for i in range(NUM_PRODUCTS):
        product = {
            'product_id': f"prod_{i:05d}",
            'product_name': f"Product {i}",
            'category': random.choice(generator.CATEGORIES),
            'price': round(random.uniform(9.99, 999.99), 2),
            'stock_quantity': random.randint(0, 1000),
            'rating': round(random.uniform(1, 5), 1),
            'num_reviews': random.randint(0, 1000)
        }
        products.append(product)

    # Save product catalog
    with open(base_path / 'product_catalog.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)

    print(f"Saved {len(products)} products to product_catalog.csv")
    print("\nData generation complete!")


if __name__ == "__main__":
    main()
