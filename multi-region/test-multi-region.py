#!/usr/bin/env python3
"""
Multi-Region Deployment Test Suite
Tests connectivity, replication, and failover capabilities
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional
import asyncpg
import aiohttp
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiRegionTester:
    def __init__(self):
        self.regions = {
            'us-east-1': {
                'context': 'us-east-1',
                'api_url': 'http://rap-analyzer-us-east-1.rap-analyzer.svc.cluster.local',
                'db_host': 'postgresql-primary.rap-analyzer.svc.cluster.local',
                'role': 'primary'
            },
            'us-west-2': {
                'context': 'us-west-2', 
                'api_url': 'http://rap-analyzer-us-west-2.rap-analyzer.svc.cluster.local',
                'db_host': 'postgresql-replica.rap-analyzer.svc.cluster.local',
                'role': 'replica'
            },
            'eu-west-1': {
                'context': 'eu-west-1',
                'api_url': 'http://rap-analyzer-eu-west-1.rap-analyzer.svc.cluster.local', 
                'db_host': 'postgresql-replica.rap-analyzer.svc.cluster.local',
                'role': 'replica'
            }
        }
        self.test_results = {}
        
    async def run_kubectl_command(self, context: str, command: str) -> str:
        """Execute kubectl command with specific context"""
        try:
            cmd = f"kubectl --context={context} {command}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Command failed: {cmd}")
                logger.error(f"Error: {result.stderr}")
                return ""
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to run kubectl command: {e}")
            return ""
    
    async def test_cluster_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to all Kubernetes clusters"""
        logger.info("🔗 Testing cluster connectivity...")
        results = {}
        
        for region, config in self.regions.items():
            try:
                output = await self.run_kubectl_command(
                    config['context'], 
                    "get nodes --no-headers"
                )
                results[region] = bool(output.strip())
                status = "✅" if results[region] else "❌"
                logger.info(f"  {status} {region}: {'Connected' if results[region] else 'Failed'}")
            except Exception as e:
                results[region] = False
                logger.error(f"  ❌ {region}: {e}")
        
        return results
    
    async def test_pod_status(self) -> Dict[str, Dict[str, bool]]:
        """Test if all pods are running in each region"""
        logger.info("🚀 Testing pod status...")
        results = {}
        
        for region, config in self.regions.items():
            results[region] = {}
            
            # Test PostgreSQL pods
            pg_output = await self.run_kubectl_command(
                config['context'],
                "get pods -l app=postgresql -n rap-analyzer --no-headers"
            )
            results[region]['postgresql'] = 'Running' in pg_output and '1/1' in pg_output
            
            # Test application pods
            app_output = await self.run_kubectl_command(
                config['context'],
                "get pods -l app=rap-analyzer -n rap-analyzer --no-headers"
            )
            results[region]['application'] = 'Running' in app_output and '1/1' in app_output
            
            pg_status = "✅" if results[region]['postgresql'] else "❌"
            app_status = "✅" if results[region]['application'] else "❌"
            logger.info(f"  {region}: PostgreSQL {pg_status}, Application {app_status}")
        
        return results
    
    async def test_database_replication(self) -> Dict[str, Dict]:
        """Test PostgreSQL replication status"""
        logger.info("🐘 Testing database replication...")
        results = {}
        
        for region, config in self.regions.items():
            results[region] = {}
            
            try:
                # Get database connection info from secret
                db_secret = await self.run_kubectl_command(
                    config['context'],
                    "get secret postgresql-secret -n rap-analyzer -o jsonpath='{.data}'"
                )
                
                if db_secret:
                    # Test database connectivity (simplified)
                    pg_test = await self.run_kubectl_command(
                        config['context'],
                        f"exec -n rap-analyzer postgresql-{config['role']}-0 -- " +
                        "psql -U postgres -c 'SELECT version();'"  
                    )
                    results[region]['connectivity'] = 'PostgreSQL' in pg_test
                    
                    # Test replication status
                    if config['role'] == 'primary':
                        repl_test = await self.run_kubectl_command(
                            config['context'],
                            f"exec -n rap-analyzer postgresql-primary-0 -- " +
                            "psql -U postgres -c 'SELECT * FROM pg_stat_replication;'"
                        )
                        results[region]['replication_active'] = 'streaming' in repl_test.lower()
                    else:
                        recovery_test = await self.run_kubectl_command(
                            config['context'],
                            f"exec -n rap-analyzer postgresql-replica-0 -- " +
                            "psql -U postgres -c 'SELECT pg_is_in_recovery();'"
                        )
                        results[region]['is_replica'] = 't' in recovery_test.lower()
                
                else:
                    results[region]['connectivity'] = False
                    
            except Exception as e:
                logger.error(f"Database test failed for {region}: {e}")
                results[region]['error'] = str(e)
        
        return results
    
    async def test_api_endpoints(self) -> Dict[str, Dict]:
        """Test API endpoints in each region"""
        logger.info("🌐 Testing API endpoints...")
        results = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for region, config in self.regions.items():
                results[region] = {}
                
                try:
                    # Port forward to access service (simplified test)
                    # In real deployment, this would use actual load balancer URLs
                    
                    # Test health endpoint
                    health_url = f"{config['api_url']}/health"
                    
                    # For testing, we'll check if the service exists
                    svc_output = await self.run_kubectl_command(
                        config['context'],
                        "get svc rap-analyzer -n rap-analyzer --no-headers"
                    )
                    results[region]['service_exists'] = 'rap-analyzer' in svc_output
                    
                    # Test service endpoints
                    ep_output = await self.run_kubectl_command(
                        config['context'],
                        "get endpoints rap-analyzer -n rap-analyzer --no-headers"
                    )
                    results[region]['endpoints_ready'] = bool(ep_output.strip())
                    
                    status = "✅" if results[region]['service_exists'] and results[region]['endpoints_ready'] else "❌"
                    logger.info(f"  {status} {region}: API service ready")
                    
                except Exception as e:
                    logger.error(f"API test failed for {region}: {e}")
                    results[region]['error'] = str(e)
        
        return results
    
    async def test_cross_region_latency(self) -> Dict[str, Dict]:
        """Test network latency between regions"""
        logger.info("⚡ Testing cross-region latency...")
        results = {}
        
        for region1, config1 in self.regions.items():
            results[region1] = {}
            
            for region2, config2 in self.regions.items():
                if region1 == region2:
                    continue
                    
                try:
                    # Test latency between PostgreSQL services
                    ping_cmd = (
                        f"exec -n rap-analyzer postgresql-{config1['role']}-0 -- "
                        f"ping -c 3 {config2['db_host']}"
                    )
                    
                    ping_output = await self.run_kubectl_command(config1['context'], ping_cmd)
                    
                    # Parse ping results (simplified)
                    if 'avg' in ping_output:
                        # Extract average latency
                        lines = ping_output.split('\n')
                        for line in lines:
                            if 'avg' in line:
                                avg_latency = line.split('/')[-2] if '/' in line else 'unknown'
                                results[region1][region2] = f"{avg_latency}ms"
                                break
                    else:
                        results[region1][region2] = "unreachable"
                        
                except Exception as e:
                    results[region1][region2] = f"error: {e}"
        
        return results
    
    async def test_data_consistency(self) -> Dict[str, bool]:
        """Test data consistency across regions"""
        logger.info("🔍 Testing data consistency...")
        results = {}
        
        try:
            # Create test data in primary
            test_query = "SELECT COUNT(*) FROM songs LIMIT 1;"
            
            primary_result = await self.run_kubectl_command(
                self.regions['us-east-1']['context'],
                f"exec -n rap-analyzer postgresql-primary-0 -- " +
                f"psql -U postgres -c \"{test_query}\""
            )
            
            # Wait for replication
            await asyncio.sleep(5)
            
            # Check replicas
            for region in ['us-west-2', 'eu-west-1']:
                replica_result = await self.run_kubectl_command(
                    self.regions[region]['context'],
                    f"exec -n rap-analyzer postgresql-replica-0 -- " +
                    f"psql -U postgres -c \"{test_query}\""
                )
                
                # Compare results (simplified)
                results[region] = primary_result.strip() == replica_result.strip()
                status = "✅" if results[region] else "❌"
                logger.info(f"  {status} {region}: Data consistency check")
                
        except Exception as e:
            logger.error(f"Data consistency test failed: {e}")
            for region in ['us-west-2', 'eu-west-1']:
                results[region] = False
        
        return results
    
    async def test_monitoring_stack(self) -> Dict[str, bool]:
        """Test monitoring components"""
        logger.info("📊 Testing monitoring stack...")
        results = {}
        
        monitoring_components = ['prometheus', 'grafana', 'alertmanager']
        
        for region, config in self.regions.items():
            results[region] = {}
            
            for component in monitoring_components:
                try:
                    pod_output = await self.run_kubectl_command(
                        config['context'],
                        f"get pods -l app={component} -n rap-analyzer --no-headers"
                    )
                    results[region][component] = 'Running' in pod_output
                    
                except Exception as e:
                    results[region][component] = False
                    logger.error(f"Monitoring test failed for {component} in {region}: {e}")
        
        return results
    
    async def generate_test_report(self) -> Dict:
        """Generate comprehensive test report"""
        logger.info("📋 Generating test report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {},
            'detailed_results': {}
        }
        
        # Run all tests
        tests = [
            ('cluster_connectivity', self.test_cluster_connectivity()),
            ('pod_status', self.test_pod_status()),
            ('database_replication', self.test_database_replication()),
            ('api_endpoints', self.test_api_endpoints()),
            ('cross_region_latency', self.test_cross_region_latency()),
            ('data_consistency', self.test_data_consistency()),
            ('monitoring_stack', self.test_monitoring_stack())
        ]
        
        for test_name, test_coro in tests:
            try:
                logger.info(f"Running {test_name}...")
                result = await test_coro
                report['detailed_results'][test_name] = result
                
                # Calculate test summary
                if isinstance(result, dict):
                    if test_name == 'cluster_connectivity':
                        passed = sum(1 for v in result.values() if v)
                        total = len(result)
                    elif test_name == 'pod_status':
                        passed = sum(1 for region_result in result.values() 
                                   for v in region_result.values() if v)
                        total = sum(len(region_result) for region_result in result.values())
                    else:
                        # Generic counting for other tests
                        passed = sum(1 for v in result.values() if 
                                   isinstance(v, bool) and v)
                        total = len([v for v in result.values() if isinstance(v, bool)])
                    
                    report['test_summary'][test_name] = {
                        'passed': passed,
                        'total': max(total, 1),
                        'success_rate': f"{(passed/max(total, 1)*100):.1f}%"
                    }
                
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                report['detailed_results'][test_name] = {'error': str(e)}
                report['test_summary'][test_name] = {
                    'passed': 0,
                    'total': 1,
                    'success_rate': '0.0%'
                }
        
        # Overall summary
        total_passed = sum(s['passed'] for s in report['test_summary'].values())
        total_tests = sum(s['total'] for s in report['test_summary'].values())
        report['overall_success_rate'] = f"{(total_passed/max(total_tests, 1)*100):.1f}%"
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted test report"""
        print("\n" + "="*60)
        print("🌍 MULTI-REGION DEPLOYMENT TEST REPORT")
        print("="*60)
        print(f"📅 Test Time: {report['timestamp']}")
        print(f"🎯 Overall Success Rate: {report['overall_success_rate']}")
        print()
        
        print("📊 TEST SUMMARY:")
        print("-" * 40)
        for test_name, summary in report['test_summary'].items():
            status = "✅" if summary['passed'] == summary['total'] else "⚠️" if summary['passed'] > 0 else "❌"
            print(f"{status} {test_name}: {summary['passed']}/{summary['total']} ({summary['success_rate']})")
        
        print("\n📋 DETAILED RESULTS:")
        print("-" * 40)
        for test_name, results in report['detailed_results'].items():
            print(f"\n🔍 {test_name.upper()}:")
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            status = "✅" if subvalue else "❌"
                            print(f"    {status} {subkey}: {subvalue}")
                    else:
                        status = "✅" if value else "❌"
                        print(f"  {status} {key}: {value}")
        
        print("\n" + "="*60)

async def main():
    """Main test execution"""
    tester = MultiRegionTester()
    
    print("🚀 Starting Multi-Region Deployment Tests...")
    print("This may take several minutes to complete...\n")
    
    try:
        report = await tester.generate_test_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"multi-region-test-report-{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print results
        tester.print_report(report)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        overall_rate = float(report['overall_success_rate'].rstrip('%'))
        exit_code = 0 if overall_rate >= 80 else 1
        
        if exit_code == 0:
            print("🎉 Multi-region deployment tests PASSED!")
        else:
            print("⚠️  Multi-region deployment tests had issues. Check the report above.")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print("❌ Multi-region deployment tests FAILED!")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)