"""Command-line interface for mining segmentation downloader."""

import argparse
import logging
from pathlib import Path

from .database import DownloadDatabase
from .config import Config
from .tasks import create_tasks

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration.
    
    Args:
        verbose: Enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Download Landsat data for mining regions'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create tasks command
    create_parser = subparsers.add_parser('create', help='Create download tasks')
    create_parser.add_argument(
        '--mining-file',
        type=str,
        required=True,
        help='Path to mining polygons GeoPackage file'
    )
    create_parser.add_argument(
        '--countries',
        type=str,
        nargs='+',
        help='ISO3 country codes (e.g., ZAF USA). If not specified, all countries.'
    )
    create_parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        required=True,
        help='Years to download (e.g., 2020 2021 2022)'
    )
    create_parser.add_argument(
        '--year-range',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Year range (e.g., 1992 2021)'
    )
    create_parser.add_argument(
        '--buffer',
        type=float,
        default=0.05,
        help='Buffer size in degrees around each mine for clustering (default: 0.05)'
    )
    
    # Run workers command
    run_parser = subparsers.add_parser('run', help='Run workers')
    run_parser.add_argument(
        '--workers',
        type=str,
        nargs='+',
        choices=['export', 'status', 'download', 'reproject', 'transfer', 'janitor', 'all'],
        default=['all'],
        help='Workers to run (default: all)'
    )
    run_parser.add_argument(
        '--countries',
        type=str,
        nargs='+',
        help='ISO3 country codes to filter tasks (e.g., ZAF USA). If not specified, all countries.'
    )
    run_parser.add_argument(
        '--once',
        action='store_true',
        help='Run once instead of continuously'
    )
    run_parser.add_argument(
        '--clean',
        action='store_true',
        help='Enable cleanup mode for janitor (removes stale file references)'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show download status')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup database to HPC')
    
    # Removed compress-remaining command (no longer needed with zarr)
    
    # Common arguments
    for p in [create_parser, run_parser, status_parser, backup_parser]:
        p.add_argument(
            '--db',
            type=str,
            default=None,
            help='Database path (default: from config)'
        )
        p.add_argument(
            '--verbose',
            '-v',
            action='store_true',
            help='Enable verbose logging'
        )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    # Initialize config and database
    config = Config()
    config.ensure_dirs()
    
    db_path = args.db if hasattr(args, 'db') and args.db else config.DB_PATH
    db = DownloadDatabase(str(db_path))
    
    # Execute command
    if args.command == 'create':
        # Determine years
        if args.year_range:
            years = list(range(args.year_range[0], args.year_range[1] + 1))
        else:
            years = args.years
        
        create_tasks(
            args.mining_file,
            args.countries,
            years,
            args.buffer,
            db
        )
    
    elif args.command == 'run':
        from .gee_export import GEEExportWorker
        from .status_checker import StatusCheckerWorker
        from .download import DownloadWorker
        from .reproject import ReprojectionWorker
        from .transfer import TransferWorker
        from .janitor import JanitorWorker
        import threading
        
        continuous = not args.once
        workers_to_run = args.workers
        countries = args.countries if hasattr(args, 'countries') else None
        
        if 'all' in workers_to_run:
            workers_to_run = ['export', 'status', 'download', 'reproject', 'transfer', 'janitor']
        
        if countries:
            logger.info(f"Filtering tasks for countries: {', '.join(countries)}")
        
        threads = []
        
        if 'export' in workers_to_run:
            logger.info("Starting export worker")
            worker = GEEExportWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'status' in workers_to_run:
            logger.info("Starting status checker worker")
            worker = StatusCheckerWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'download' in workers_to_run:
            logger.info("Starting download worker")
            worker = DownloadWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'reproject' in workers_to_run:
            logger.info("Starting reprojection worker")
            worker = ReprojectionWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'transfer' in workers_to_run:
            logger.info("Starting transfer worker")
            worker = TransferWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'janitor' in workers_to_run:
            logger.info("Starting janitor worker")
            clean_mode = args.clean if hasattr(args, 'clean') else False
            worker = JanitorWorker(db, config, countries=countries, clean=clean_mode)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
    
    elif args.command == 'status':
        stats = db.get_statistics()
        
        print("\n=== Download Statistics ===")
        print(f"\nTotal tasks: {stats['total_tasks']}")
        
        print("\nStatus breakdown:")
        for status, count in stats['status_counts'].items():
            print(f"  {status}: {count}")
        
        print("\nCountries:")
        for country, count in stats['country_counts'].items():
            print(f"  {country}: {count} polygons")
        
        if stats['year_range']['min_year']:
            print(f"\nYear range: {stats['year_range']['min_year']} - {stats['year_range']['max_year']}")
    
    elif args.command == 'backup':
        from .transfer import TransferWorker
        
        logger.info("Starting database backup")
        worker = TransferWorker(db, config)
        success = worker.backup_database()
        
        if success:
            print("\n✓ Database backed up successfully to HPC")
        else:
            print("\n✗ Database backup failed - check logs")
            exit(1)


if __name__ == '__main__':
    main()
