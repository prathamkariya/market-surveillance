import pytest
from app.services.anomaly_service import score_live_trade

def test_score_live_trade_schema_includes_volume():
    """Ensure that score_live_trade always includes the volume in its output."""
    trade = {
        "event_id": "TEST_BTCUSDT_1718000000000",
        "symbol": "BTCUSDT",
        "timestamp_ms": 1718000000000,
        "price": 60000.0,
        "volume": 2.5,
    }
    
    # Empty history will trigger basic anomaly rules or return None if not anomalous.
    # We'll mock the internal _combine_scores if needed, but if the threshold is high,
    # we might just get None. Let's force an anomaly by providing a feature row 
    # that fails isolation forest, or we can just mock get_model_registry.
    
    # Actually, a simpler unit test: we can mock get_model_registry to return a dummy 
    # that always predicts anomaly, but since we just want to test the payload shape,
    # let's mock the anomaly threshold.
    
    import app.services.anomaly_service as anomaly_service
    original_threshold = anomaly_service.threshold
    
    try:
        anomaly_service.threshold = -1.0 # Guarantee everything is an anomaly
        
        # We need mock models
        class MockRegistry:
            has_any_model = False
            
        anomaly_service.get_model_registry = lambda: MockRegistry()
        
        # Wait, if has_any_model is False, it returns None.
        # Let's mock the models.
        class MockIF:
            def score_samples(self, X):
                return [0.5]
                
        class MockRegistryTrue:
            has_any_model = True
            has_isolation_forest = True
            has_multi_pattern = False
            isolation_forest = MockIF()
            
        anomaly_service.get_model_registry = lambda: MockRegistryTrue()
        
        history = [
            {"price": 59000.0, "volume": 1.0, "timestamp_ms": 1717999990000},
            {"price": 59500.0, "volume": 1.0, "timestamp_ms": 1717999995000}
        ]
        
        alert = score_live_trade(trade, history, sentiment_score=0.2)
        
        assert alert is not None
        assert alert["event_id"] == "TEST_BTCUSDT_1718000000000"
        assert alert["symbol"] == "BTCUSDT"
        assert alert["volume"] == 2.5
        assert alert["sentiment_score"] == 0.2
        assert alert["isolation_forest_score"] == 0.5
        assert "price" in alert
        
    finally:
        # Restore
        anomaly_service.threshold = original_threshold
