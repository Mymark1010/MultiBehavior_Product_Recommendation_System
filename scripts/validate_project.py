"""Run tests and repository checks, optionally refreshing full-data readiness evidence."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-data',action='store_true',help='Verify retained raw/Parquet and independently audit real next-step samples')
    parser.add_argument('--notebooks',action='store_true',help='Execute all read-only notebooks (requires optional notebook dependencies)')
    args=parser.parse_args()
    if args.notebooks and not args.full_data:
        parser.error('--notebooks requires --full-data')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.testsRun or not result.wasSuccessful():
        raise SystemExit(1)
    from scripts.check_repository import main as check
    check()
    if not args.full_data:
        return
    from scripts.verify_artifacts import verify
    from scripts.audit_nextstep import audit
    manifest=verify(ROOT)
    analysis=audit(ROOT)
    executed=[]
    if args.notebooks:
        from scripts.execute_notebooks import main as execute
        execute()
        executed=json.loads((ROOT/'reports/nextstep/notebook_validation.json').read_text())['notebooks_executed']
    summary=json.loads((ROOT/'reports/nextstep/summary.json').read_text())
    evidence={'date':datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).date().isoformat(),'task':'next_step','schema_version':'2.0.0',
              'scope':'full_retailrocket_dataset','unit_tests_run':result.testsRun,
              'unit_tests_passed':result.testsRun-len(result.skipped),
              'failed':len(result.failures)+len(result.errors),'skipped':len(result.skipped),
              'full_dataset_verified':True,'parquet_artifacts_verified':len(manifest['artifacts']),
              'pipeline_checks_passed':len(summary['checks']),
              'independent_audit_queries':analysis['independent_sample_queries'],
              'independent_audit_target_pairs':analysis['independent_sample_target_pairs'],
              'pipeline_sha256':manifest['pipeline_sha256'],'reader_sha256':manifest['reader_sha256'],
              'test_suite_sha256':hashlib.file_digest((ROOT/'tests/test_pipeline.py').open('rb'),'sha256').hexdigest(),
              'audit_sha256':hashlib.file_digest((ROOT/'scripts/audit_nextstep.py').open('rb'),'sha256').hexdigest(),
              'notebooks_executed':executed,'all_notebook_cells_completed':bool(executed),
              'ranking_models_trained':False,'demo_implemented':False}
    (ROOT/'reports/nextstep/validation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Saved full next-step validation evidence.')


if __name__=='__main__':
    main()
