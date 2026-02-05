# 🧪 TrendTap User Testing Implementation

This directory contains the complete implementation of the TrendTap User Testing Plan, providing comprehensive automated testing for all core functionalities.

## 📁 Directory Structure

```
user_testing/
├── README.md                           # This file
├── test_framework.py                   # Core testing framework
├── test_affiliate_research.py          # Affiliate research tests (AR-001 to AR-007)
├── test_trend_analysis.py              # Trend analysis tests (TA-001 to TA-007)
├── test_content_generation.py          # Content generation tests (CI-001 to CI-007)
├── test_e2e_integration.py             # End-to-end integration tests (E2E-001 to E2E-003)
├── test_performance_errors.py          # Performance & error handling tests (PERF-001 to ERR-003)
├── run_all_tests.py                    # Main test runner
├── quick_test.py                       # Quick system verification
└── test_data/                          # Test reports and data (created automatically)
```

## 🚀 Quick Start

### Prerequisites

1. **System Running**: Ensure TrendTap backend and frontend are running
   ```bash
   # Backend (Terminal 1)
   cd trend-analysis-platform/backend
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   
   # Frontend (Terminal 2)
   cd trend-analysis-platform/frontend
   npm run dev
   ```

2. **Dependencies**: Install required Python packages
   ```bash
   cd trend-analysis-platform/backend
   pip install httpx asyncio
   ```

### Running Tests

#### 1. Quick System Test
```bash
cd trend-analysis-platform/backend/tests/user_testing
python quick_test.py
```

#### 2. Run All Tests
```bash
cd trend-analysis-platform/backend/tests/user_testing
python run_all_tests.py
```

#### 3. Run Specific Test Suite
```bash
# Affiliate Research Tests
python run_all_tests.py --suite affiliate

# Trend Analysis Tests
python run_all_tests.py --suite trend

# Content Generation Tests
python run_all_tests.py --suite content

# E2E Integration Tests
python run_all_tests.py --suite e2e

# Performance & Error Tests
python run_all_tests.py --suite performance
```

## 📋 Test Coverage

### ✅ Affiliate Research Tests (AR-001 to AR-007)
- **AR-001**: Basic Affiliate Search
- **AR-002**: Program Selection & Content Generation
- **AR-003**: Advanced Search Filters
- **AR-004**: Program Details & Comparison
- **AR-005**: Invalid Search Terms
- **AR-006**: Network API Failures
- **AR-007**: Load Testing

### ✅ Trend Analysis Tests (TA-001 to TA-007)
- **TA-001**: Basic Trend Analysis
- **TA-002**: Multiple Keywords Analysis
- **TA-003**: CSV Upload for Trends
- **TA-004**: LLM Fallback Analysis
- **TA-005**: Opportunity Score Validation
- **TA-006**: Geographic Analysis
- **TA-007**: Invalid Keywords

### ✅ Content Generation Tests (CI-001 to CI-007)
- **CI-001**: Basic Content Generation
- **CI-002**: Software Solutions Generation
- **CI-003**: Content Optimization
- **CI-004**: Content Export
- **CI-005**: Content Calendar Integration
- **CI-006**: Content Collaboration
- **CI-007**: Content Generation Failures

### ✅ E2E Integration Tests (E2E-001 to E2E-003)
- **E2E-001**: Complete Research Workflow
- **E2E-002**: Multi-User Workflow
- **E2E-003**: Cross-Feature Data Flow

### ✅ Performance & Error Tests (PERF-001 to ERR-003)
- **PERF-001**: API Response Times
- **PERF-002**: Concurrent User Performance
- **ERR-001**: API Timeout Handling
- **ERR-002**: Database Connection Issues
- **ERR-003**: Malicious Input Handling

## 📊 Test Results

### Success Criteria
- **Overall Pass Rate**: ≥ 95% (Excellent), ≥ 80% (Good)
- **API Response Times**: 
  - Affiliate search: < 5 seconds
  - Trend analysis: < 15 seconds
  - Content generation: < 20 seconds
  - Keyword analysis: < 10 seconds
- **Concurrent Users**: Support 10+ concurrent users
- **Error Handling**: Graceful degradation for all error scenarios

### Test Reports
Test reports are automatically saved to `test_data/` directory with timestamps:
- `trendtap_user_testing_report_YYYYMMDD_HHMMSS.json`
- `user_testing.log` (detailed execution log)

## 🔧 Configuration

### Environment Variables
The tests use the following default configurations:
- **Backend URL**: `http://localhost:8000`
- **Frontend URL**: `http://localhost:3000`
- **Test User**: `test@trendtap.com` / `testpassword123`

### Custom Configuration
```bash
# Custom backend URL
python run_all_tests.py --base-url http://localhost:8080

# Custom frontend URL
python run_all_tests.py --frontend-url http://localhost:3001
```

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Failed**
   ```
   ❌ Authentication failed - please check if the system is running
   ```
   - Ensure backend is running on port 8000
   - Check if test user exists in database
   - Verify Supabase connection

2. **API Endpoints Not Found (404)**
   ```
   Expected 200, got 404
   ```
   - Some endpoints may not be implemented yet
   - Tests will log warnings for missing endpoints
   - This is expected for features still in development

3. **Timeout Errors**
   ```
   Request timeout
   ```
   - Check system performance
   - Verify external API connections
   - Increase timeout values if needed

4. **Database Connection Issues**
   ```
   Database connection failed
   ```
   - Verify Supabase connection
   - Check database credentials
   - Ensure database is accessible

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Monitoring

The test framework includes built-in performance monitoring:
- Response time tracking
- Concurrent user simulation
- Memory usage monitoring
- Error rate tracking

## 🔄 Continuous Integration

To integrate with CI/CD pipelines:

```bash
# Run tests and exit with appropriate code
python run_all_tests.py
echo $?  # 0 = success, 1 = failure
```

## 📝 Test Data

The framework includes realistic test data:
- **Affiliate Programs**: Sample programs from major networks
- **Trend Keywords**: Real-world trending topics
- **Content Topics**: SEO-friendly content ideas
- **CSV Files**: Sample trend data for upload testing

## 🎯 Next Steps

1. **Run Initial Tests**: Execute `quick_test.py` to verify system status
2. **Full Test Suite**: Run `run_all_tests.py` for comprehensive testing
3. **Review Results**: Check test reports for any failures
4. **Fix Issues**: Address any critical failures identified
5. **Re-run Tests**: Verify fixes with subsequent test runs

## 📞 Support

For issues or questions:
1. Check the test logs in `user_testing.log`
2. Review the detailed test report JSON files
3. Verify system status with health checks
4. Check backend logs for API errors

---

**Test Plan Status**: ✅ Implemented and Ready for Execution  
**Last Updated**: 2025-01-27  
**Version**: 1.0.0


