/**
 * Test script to verify frontend can connect to backend API
 * Run with: node test_api_connection.js
 */

const API_BASE_URL = 'http://localhost:8000';

async function testEndpoint(name, url, options = {}) {
  try {
    console.log(`\n Testing ${name}...`);
    const response = await fetch(url, options);
    const data = await response.json();
    
    if (response.ok) {
      console.log(`✅ ${name} - SUCCESS`);
      console.log(`   Status: ${response.status}`);
      console.log(`   Data:`, JSON.stringify(data, null, 2).substring(0, 200));
      return true;
    } else {
      console.log(`❌ ${name} - FAILED`);
      console.log(`   Status: ${response.status}`);
      console.log(`   Error:`, data);
      return false;
    }
  } catch (error) {
    console.log(`❌ ${name} - ERROR`);
    console.log(`   ${error.message}`);
    return false;
  }
}

async function runTests() {
  console.log('='.repeat(60));
  console.log('FRONTEND-BACKEND API CONNECTION TEST');
  console.log('='.repeat(60));
  console.log(`\nBackend URL: ${API_BASE_URL}`);
  
  const results = [];

  // Test 1: Health Check
  results.push(await testEndpoint(
    'Health Check',
    `${API_BASE_URL}/health`
  ));

  // Test 2: Root Endpoint
  results.push(await testEndpoint(
    'Root Endpoint',
    `${API_BASE_URL}/`
  ));

  // Test 3: Model Status
  results.push(await testEndpoint(
    'Model Status',
    `${API_BASE_URL}/api/v1/fraud/models/status`
  ));

  // Test 4: Dashboard Analytics
  results.push(await testEndpoint(
    'Dashboard Analytics',
    `${API_BASE_URL}/api/v1/analytics/dashboard`
  ));

  // Test 5: Fraud Prediction (with sample data)
  const sampleTransaction = {
    time: Date.now() / 1000,
    amount: 5000.0,
    v1: 0.5, v2: -1.2, v3: 0.8, v4: -0.3, v5: 1.1,
    v6: 0.2, v7: -0.7, v8: 0.4, v9: -0.9, v10: 0.6,
    v11: -0.4, v12: 0.9, v13: -0.2, v14: 0.7, v15: -0.5,
    v16: 0.3, v17: -0.8, v18: 0.1, v19: -0.6, v20: 0.8,
    v21: -0.3, v22: 0.5, v23: -0.7, v24: 0.2, v25: -0.4,
    v26: 0.6, v27: -0.1, v28: 0.4
  };

  results.push(await testEndpoint(
    'Fraud Prediction',
    `${API_BASE_URL}/api/v1/fraud/predict`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sampleTransaction)
    }
  ));

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('TEST SUMMARY');
  console.log('='.repeat(60));
  
  const passed = results.filter(r => r).length;
  const total = results.length;
  
  console.log(`\nPassed: ${passed}/${total} (${Math.round(passed/total*100)}%)`);
  
  if (passed === total) {
    console.log('\n✅ ALL TESTS PASSED - Frontend can connect to backend!');
  } else if (passed > 0) {
    console.log('\n⚠️  SOME TESTS FAILED - Check backend is running');
  } else {
    console.log('\n❌ ALL TESTS FAILED - Backend may not be running');
    console.log('\nTo start backend:');
    console.log('  cd backend');
    console.log('  .\\venv\\Scripts\\activate.bat');
    console.log('  python run_server.py');
  }
  
  console.log('\n');
}

runTests();
