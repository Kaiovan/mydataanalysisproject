"""
Unit tests for data generation module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_generation.clickstream_generator import ClickstreamGenerator


class TestClickstreamGenerator:
    """Test suite for ClickstreamGenerator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.generator = ClickstreamGenerator(num_users=10, num_products=20)

    def test_initialization(self):
        """Test generator initialization"""
        assert len(self.generator.user_ids) == 10
        assert len(self.generator.product_ids) == 20
        assert self.generator.num_users == 10
        assert self.generator.num_products == 20

    def test_generate_session(self):
        """Test session generation"""
        from datetime import datetime

        user_id = self.generator.user_ids[0]
        session_start = datetime.now()

        events = self.generator.generate_session(user_id, session_start)

        # Check that events were generated
        assert len(events) > 0
        assert all(e['user_id'] == user_id for e in events)
        assert all('session_id' in e for e in events)
        assert all('event_type' in e for e in events)
        assert all('timestamp' in e for e in events)

    def test_generate_clickstream_data(self):
        """Test full clickstream data generation"""
        events = self.generator.generate_clickstream_data(num_sessions=5, days=1)

        # Check that events were generated
        assert len(events) > 0

        # Check all events have required fields
        required_fields = ['event_id', 'session_id', 'user_id', 'timestamp', 'event_type']
        for event in events:
            for field in required_fields:
                assert field in event

    def test_ip_generation(self):
        """Test IP address generation"""
        ip = self.generator._generate_ip()

        # Check IP format
        parts = ip.split('.')
        assert len(parts) == 4
        assert all(0 <= int(p) <= 255 for p in parts)

    def test_event_types(self):
        """Test that generated events have valid event types"""
        events = self.generator.generate_clickstream_data(num_sessions=10, days=1)

        event_types = set(e['event_type'] for e in events)
        valid_event_types = set(self.generator.EVENT_TYPES.keys())

        # All event types should be valid
        assert event_types.issubset(valid_event_types)


class TestDataSaving:
    """Test data saving functionality"""

    def test_save_to_csv(self, tmp_path):
        """Test saving to CSV"""
        generator = ClickstreamGenerator(num_users=5, num_products=10)
        events = generator.generate_clickstream_data(num_sessions=3, days=1)

        output_file = tmp_path / "test_events.csv"
        generator.save_to_csv(events, str(output_file))

        # Check file exists
        assert output_file.exists()

        # Check file has content
        assert output_file.stat().st_size > 0

    def test_save_to_jsonl(self, tmp_path):
        """Test saving to JSONL"""
        generator = ClickstreamGenerator(num_users=5, num_products=10)
        events = generator.generate_clickstream_data(num_sessions=3, days=1)

        output_file = tmp_path / "test_events.jsonl"
        generator.save_to_jsonl(events, str(output_file))

        # Check file exists
        assert output_file.exists()

        # Check file has content
        assert output_file.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
