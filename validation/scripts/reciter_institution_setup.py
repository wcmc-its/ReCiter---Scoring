#!/usr/bin/env python3
"""ReCiter Institution Setup — discover configuration from PubMed.

CLI tool that generates ReCiter configuration for a new institution.
Uses only public PubMed data — no gold standard, no institutional data needed.

Usage (non-interactive):
  python3 scripts/reciter_institution_setup.py \\
    --domain fredhutch.org \\
    --institution-name "Fred Hutchinson Cancer Center" \\
    --year-range 2020:2025 \\
    --non-interactive --top-n 10

Usage (interactive):
  python3 scripts/reciter_institution_setup.py

The script will:
  1. Accept institution parameters via CLI flags or interactive prompts
  2. Search PubMed for recent articles with the email domain
  3. Mine affiliation strings to discover:
     - Home institution keyword patterns
     - Collaborating institution patterns
     - Alternate email domains
  4. Auto-select or present findings for user confirmation
  5. Output application.properties and SPRING_APPLICATION_JSON config
     with Scopus always disabled for external validation
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree


PUBMED_ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
PUBMED_EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
PUBMED_API_KEY = os.environ.get('PUBMED_API_KEY')  # Set via env var for higher rate limits

# Scopus is always disabled for external validation — external sites do not
# have Scopus API keys and ReCiter should not attempt Scopus retrieval.
SCOPUS_DISABLED_CONFIG = {
    'use.scopus.articles': 'false',
    'strategy.scopus.common.affiliation': 'false',
}


def pubmed_search(query: str, retmax: int = 500) -> list:
    """Search PubMed and return list of PMIDs."""
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': retmax,
        'retmode': 'json',
        'sort': 'date',
    }
    if PUBMED_API_KEY:
        params['api_key'] = PUBMED_API_KEY

    url = f'{PUBMED_ESEARCH}?{urllib.parse.urlencode(params)}'
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    pmids = data.get('esearchresult', {}).get('idlist', [])
    total = int(data.get('esearchresult', {}).get('count', 0))
    return pmids, total


def pubmed_fetch_articles(pmids: list, batch_size: int = 100) -> list:
    """Fetch article XML from PubMed and extract author affiliations."""
    articles = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {
            'db': 'pubmed',
            'id': ','.join(batch),
            'retmode': 'xml',
        }
        if PUBMED_API_KEY:
            params['api_key'] = PUBMED_API_KEY

        url = f'{PUBMED_EFETCH}?{urllib.parse.urlencode(params)}'
        with urllib.request.urlopen(url, timeout=60) as resp:
            xml_data = resp.read()

        root = ElementTree.fromstring(xml_data)
        for art_elem in root.findall('.//PubmedArticle'):
            pmid_elem = art_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None

            authors = []
            for author in art_elem.findall('.//Author'):
                last = author.findtext('LastName', '')
                fore = author.findtext('ForeName', '')
                affiliations = []
                for aff in author.findall('.//AffiliationInfo/Affiliation'):
                    if aff.text:
                        affiliations.append(aff.text)
                authors.append({
                    'lastName': last,
                    'foreName': fore,
                    'affiliations': affiliations,
                })

            articles.append({'pmid': pmid, 'authors': authors})

        print(f'  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} articles...', flush=True)
        time.sleep(0.4)  # Rate limit

    return articles


def extract_institution_patterns(affiliations: list, email_domain: str) -> dict:
    """Analyze affiliation strings to find institution patterns."""
    # Clean the domain for matching
    domain_base = email_domain.lstrip('@').split('.')[0]  # e.g., "fredhutch"

    # Count full affiliation strings containing the domain
    home_affiliations = Counter()
    all_affiliations = Counter()
    coauthor_affiliations = Counter()

    for aff_list, is_home_author in affiliations:
        for aff in aff_list:
            aff_clean = aff.strip().rstrip('.')
            all_affiliations[aff_clean] += 1
            if is_home_author:
                home_affiliations[aff_clean] += 1
            else:
                coauthor_affiliations[aff_clean] += 1

    return {
        'home': home_affiliations,
        'coauthor': coauthor_affiliations,
        'all': all_affiliations,
    }


def extract_institution_names(affiliation_counter: Counter, top_n: int = 50) -> list:
    """Extract institution names from affiliation strings.

    Returns list of (institution_name, count) tuples sorted by frequency.
    """
    institution_counter = Counter()

    for aff_string, count in affiliation_counter.items():
        # Split by comma and look for institution-like segments
        parts = [p.strip() for p in aff_string.split(',')]
        for part in parts:
            # Skip parts that look like departments, cities, states, countries, zip codes
            part_lower = part.lower()
            if any(skip in part_lower for skip in [
                'department of', 'division of', 'section of', 'program in',
                'usa', 'united states', 'canada', 'uk', 'china', 'japan',
                'electronic address', '@',
            ]):
                continue
            if re.match(r'^\d{5}', part):  # Zip codes
                continue
            if len(part) < 5 or len(part) > 100:
                continue

            # Likely institution names contain: university, hospital, center, institute, college, school
            if any(kw in part_lower for kw in [
                'university', 'hospital', 'center', 'centre', 'institute',
                'college', 'school', 'medical', 'cancer', 'research',
                'laboratory', 'clinic', 'foundation', 'health',
            ]):
                institution_counter[part] += count

    return institution_counter.most_common(top_n)


def generate_keywords(institution_name: str) -> str:
    """Generate ReCiter-style keyword pattern from an institution name.

    ReCiter uses | for AND within a group, , for OR between groups.
    E.g., 'Fred Hutchinson Cancer Center' -> 'fred|hutchinson|cancer'
    """
    stopwords = {'of', 'the', 'for', 'and', 'to', 'in', 'at', 'a', 'an'}
    words = [w.lower() for w in institution_name.split()
             if w.lower() not in stopwords and len(w) > 1]

    if len(words) <= 1:
        return words[0] if words else ''

    # Use 2-3 most distinctive words joined by |
    # Skip very common words like "University", "Hospital" if there are more specific ones
    generic = {'university', 'hospital', 'center', 'centre', 'medical', 'school',
               'college', 'institute', 'research', 'health', 'national'}
    specific = [w for w in words if w not in generic]
    if len(specific) >= 2:
        return '|'.join(specific[:3])
    else:
        return '|'.join(words[:3])


def discover_email_domains(articles: list, primary_domain: str) -> list:
    """Discover alternate email domains from PubMed affiliation strings.

    Extracts email addresses from affiliations of authors whose affiliation
    contains the primary domain. This ensures we only discover domains that
    are co-located with the home institution (not random co-author domains).

    Args:
        articles: List of article dicts with 'authors' containing 'affiliations'.
        primary_domain: The primary email domain (e.g., 'fredhutch.org').

    Returns:
        List of (domain, count) tuples sorted by count descending.
        The primary domain is always included even if not found in affiliations.
    """
    email_pattern = re.compile(r'[\w.+-]+@([\w.-]+\.\w+)')
    domain_counts = Counter()
    primary_clean = primary_domain.lstrip('@').lower()

    for art in articles:
        for author in art.get('authors', []):
            affs = author.get('affiliations', [])
            # Check if this author is affiliated with the home institution
            is_home_author = any(primary_clean in aff.lower() for aff in affs)
            if not is_home_author:
                continue
            # Extract email domains from home-author affiliations
            for aff in affs:
                for match in email_pattern.finditer(aff):
                    domain = match.group(1).lower()
                    domain_counts[domain] += 1

    # Always include the primary domain
    if primary_clean not in domain_counts:
        domain_counts[primary_clean] = 0

    # Sort by count descending
    return domain_counts.most_common()


def auto_select_institutions(institutions: list, top_n: int = 10) -> list:
    """Auto-select the top N institutions by frequency.

    Used in non-interactive mode instead of ask_user_selection.

    Args:
        institutions: List of (name, count) tuples sorted by frequency.
        top_n: Number of institutions to select.

    Returns:
        The top top_n entries from the list.
    """
    return institutions[:top_n]


def generate_config(home_keywords_str: str, collab_keywords_str: str,
                    email_suffixes: str, institution_label: str) -> dict:
    """Generate the complete ReCiter configuration dictionary.

    Scopus is always disabled — external sites do not have Scopus API keys.

    Args:
        home_keywords_str: Comma-separated keyword groups for home institution.
        collab_keywords_str: Comma-separated keyword groups for collaborating institutions.
        email_suffixes: Comma-separated @-prefixed email domain suffixes.
        institution_label: Display name for the institution.

    Returns:
        Config dict ready for JSON or properties file output.
    """
    config = {
        'strategy.email.default.suffixes': email_suffixes,
        'strategy.authorAffiliationScoringStrategy.homeInstitution-keywords': home_keywords_str,
        'strategy.authorAffiliationScoringStrategy.homeInstitution-label': institution_label,
        'strategy.authorAffiliationScoringStrategy.collaboratingInstitutions-keywords': collab_keywords_str,
    }
    # Scopus is always disabled — this cannot be overridden
    config.update(SCOPUS_DISABLED_CONFIG)
    return config


def ask_user_selection(items: list, prompt: str, max_display: int = 20) -> list:
    """Present items to user and ask for selection."""
    print(f'\n{prompt}')
    print(f'Enter numbers to select (comma-separated), or "all" for all, "none" to skip:')
    print()
    for i, (name, count) in enumerate(items[:max_display], 1):
        kw = generate_keywords(name)
        print(f'  {i:>3}. [{count:>4}x] {name}')
        print(f'       -> keyword: {kw}')
    if len(items) > max_display:
        print(f'  ... and {len(items) - max_display} more')

    while True:
        choice = input('\nSelection: ').strip().lower()
        if choice == 'none':
            return []
        if choice == 'all':
            return items[:max_display]
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            return [items[i] for i in indices if 0 <= i < len(items)]
        except (ValueError, IndexError):
            print('Invalid input. Enter comma-separated numbers, "all", or "none".')


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='ReCiter Institution Setup - discover configuration from PubMed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Non-interactive (fully automated):
  python3 scripts/reciter_institution_setup.py \\
    --domain fredhutch.org \\
    --institution-name "Fred Hutchinson Cancer Center" \\
    --year-range 2020:2025 \\
    --non-interactive --top-n 10

  # Interactive (prompts for each step):
  python3 scripts/reciter_institution_setup.py

  # Partially automated (provide domain, prompt for institutions):
  python3 scripts/reciter_institution_setup.py \\
    --domain fredhutch.org \\
    --institution-name "Fred Hutchinson Cancer Center"
        """,
    )
    parser.add_argument('--domain', type=str, default=None,
                        help='Email domain (e.g., fredhutch.org). Skips interactive prompt.')
    parser.add_argument('--institution-name', type=str, default=None,
                        help='Institution display name (e.g., "Fred Hutchinson Cancer Center").')
    parser.add_argument('--year-range', type=str, default=None,
                        help='PubMed publication year range (e.g., 2015:2025).')
    parser.add_argument('--email-domains', type=str, default=None,
                        help='Comma-separated email domains to override/supplement auto-discovery.')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Auto-select top N institutions by frequency, no prompts.')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of institutions to auto-select in non-interactive mode (default: 10).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for config files (default: external_validation/).')
    return parser.parse_args(argv)


def main():
    args = parse_args()

    print('=' * 60)
    print(' ReCiter Institution Setup')
    print(' Discover configuration from PubMed')
    print('=' * 60)

    # Step 1: Get parameters — from CLI flags or interactive prompts
    if args.domain:
        email_domain = args.domain.strip()
    else:
        email_domain = input('\nEmail domain (e.g., @fredhutch.org): ').strip()

    if not email_domain.startswith('@'):
        email_domain = '@' + email_domain
    domain_for_search = email_domain.lstrip('@')

    if args.institution_name:
        institution_label = args.institution_name.strip()
    else:
        institution_label = input('Institution name (e.g., Fred Hutchinson Cancer Center): ').strip()

    if args.year_range:
        year_range = args.year_range.strip()
    else:
        year_range = input('Publication year range (e.g., 2015:2025, or Enter for all): ').strip()

    date_filter = f' AND {year_range}[dp]' if year_range and year_range.lower() != 'all' else ''

    # Step 2: Search PubMed
    print(f'\nSearching PubMed for articles with "{domain_for_search}" in affiliations...')
    query = f'{domain_for_search}[ad]{date_filter}'
    pmids, total = pubmed_search(query, retmax=500)
    print(f'  Found {total} total articles, fetching {len(pmids)} most recent')

    if not pmids:
        print('No articles found. Check the email domain.')
        # In non-interactive mode, still produce a minimal config
        if args.non_interactive:
            print('\nProducing minimal config with provided parameters...')
            email_suffixes = email_domain
            if args.email_domains:
                extra = ','.join(
                    d.strip() if d.strip().startswith('@') else f'@{d.strip()}'
                    for d in args.email_domains.split(',')
                )
                email_suffixes = f'{email_domain},{extra}'
            config = generate_config(
                home_keywords_str=email_domain,
                collab_keywords_str='',
                email_suffixes=email_suffixes,
                institution_label=institution_label,
            )
            _write_config(config, domain_for_search, institution_label, args.output_dir)
            return
        sys.exit(1)

    # Step 3: Fetch and parse
    print(f'\nFetching article details from PubMed...')
    articles = pubmed_fetch_articles(pmids)

    # Separate home authors (affiliation contains domain) from co-authors
    home_author_affils = Counter()
    coauthor_affils = Counter()

    for art in articles:
        for author in art['authors']:
            is_home = any(domain_for_search in aff for aff in author['affiliations'])
            for aff in author['affiliations']:
                if is_home:
                    home_author_affils[aff] += 1
                else:
                    coauthor_affils[aff] += 1

    print(f'\n  Unique home-author affiliations: {len(home_author_affils)}')
    print(f'  Unique co-author affiliations: {len(coauthor_affils)}')

    # Step 3.5: Discover email domains
    print(f'\n{"="*60}')
    print(f' EMAIL DOMAIN DISCOVERY')
    print(f'{"="*60}')

    discovered_domains = discover_email_domains(articles, domain_for_search)
    if discovered_domains:
        print(f'\nDiscovered email domains from PubMed affiliations:')
        for domain, count in discovered_domains:
            marker = ' (primary)' if domain == domain_for_search.lower() else ''
            print(f'  @{domain} ({count} occurrences){marker}')

    # Step 4: Extract home institution patterns
    print(f'\n{"="*60}')
    print(f' HOME INSTITUTION PATTERNS')
    print(f'{"="*60}')
    print(f'\nThese affiliation strings contain "{domain_for_search}":')

    home_institutions = extract_institution_names(home_author_affils)
    if not home_institutions:
        print('\nCould not extract institution names. Showing raw affiliations:')
        for aff, count in home_author_affils.most_common(10):
            print(f'  [{count}x] {aff[:120]}')

    if args.non_interactive:
        selected_home = auto_select_institutions(home_institutions, args.top_n)
        print(f'\nAuto-selected top {len(selected_home)} home institutions:')
        for name, count in selected_home:
            print(f'  [{count:>4}x] {name} -> {generate_keywords(name)}')
    else:
        selected_home = ask_user_selection(
            home_institutions,
            'Select your home institution(s) -- these become homeInstitution-keywords:',
        )

    # Step 5: Extract collaborating institutions
    print(f'\n{"="*60}')
    print(f' COLLABORATING INSTITUTIONS')
    print(f'{"="*60}')
    print(f'\nMost frequent institutions among co-authors (not {institution_label}):')

    collab_institutions = extract_institution_names(coauthor_affils)
    # Filter out the home institution
    home_words = set(institution_label.lower().split())
    collab_institutions = [
        (name, count) for name, count in collab_institutions
        if not any(w in name.lower() for w in home_words if len(w) > 3)
    ]

    if args.non_interactive:
        selected_collabs = auto_select_institutions(collab_institutions, args.top_n)
        print(f'\nAuto-selected top {len(selected_collabs)} collaborating institutions:')
        for name, count in selected_collabs:
            print(f'  [{count:>4}x] {name} -> {generate_keywords(name)}')
    else:
        selected_collabs = ask_user_selection(
            collab_institutions,
            'Select collaborating institutions -- these boost co-author affiliation matching:',
        )

    # Step 6: Build email suffixes
    # Start with discovered domains or primary domain
    if args.email_domains:
        # User-provided domains override auto-discovery
        domain_list = []
        for d in args.email_domains.split(','):
            d = d.strip()
            if not d.startswith('@'):
                d = '@' + d
            domain_list.append(d)
        # Ensure primary domain is first
        if email_domain not in domain_list:
            domain_list.insert(0, email_domain)
        email_suffixes = ','.join(domain_list)
    elif discovered_domains:
        # Use auto-discovered domains
        domain_list = [f'@{d}' for d, count in discovered_domains]
        # Ensure primary is first
        primary_at = f'@{domain_for_search.lower()}'
        if primary_at in domain_list:
            domain_list.remove(primary_at)
        domain_list.insert(0, email_domain)
        email_suffixes = ','.join(domain_list)
    else:
        email_suffixes = email_domain
        if not args.non_interactive:
            additional = input(f'\nAdditional email domains? (comma-separated, or Enter to skip): ').strip()
            if additional:
                extras = []
                for d in additional.split(','):
                    d = d.strip()
                    if not d.startswith('@'):
                        d = '@' + d
                    extras.append(d)
                email_suffixes = ','.join([email_domain] + extras)

    # Step 7: Generate configuration
    print(f'\n{"="*60}')
    print(f' GENERATED CONFIGURATION')
    print(f'{"="*60}')

    # Home institution keywords
    home_keywords = []
    for name, count in selected_home:
        kw = generate_keywords(name)
        if kw:
            home_keywords.append(kw)
    # Always add the email domain as a keyword
    home_keywords.append(email_domain)
    home_keywords_str = ', '.join(home_keywords)

    # Collaborating institution keywords
    collab_keywords = []
    for name, count in selected_collabs:
        kw = generate_keywords(name)
        if kw:
            collab_keywords.append(kw)
    collab_keywords_str = ', '.join(collab_keywords) if collab_keywords else ''

    config = generate_config(
        home_keywords_str=home_keywords_str,
        collab_keywords_str=collab_keywords_str,
        email_suffixes=email_suffixes,
        institution_label=institution_label,
    )

    _write_config(config, domain_for_search, institution_label, args.output_dir)


def _write_config(config: dict, domain_for_search: str, institution_label: str,
                  output_dir_override: str = None):
    """Write config to properties and JSON files.

    Args:
        config: Configuration dictionary.
        domain_for_search: Domain without @ (e.g., 'fredhutch.org').
        institution_label: Institution display name.
        output_dir_override: If provided, use this directory instead of default.
    """
    print(f'\n--- application.properties overrides ---\n')
    for key, value in config.items():
        print(f'{key}={value}')

    # Determine output directory
    if output_dir_override:
        output_dir = Path(output_dir_override)
    else:
        output_dir = Path(__file__).parent.parent / 'external_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_prefix = domain_for_search.split('.')[0]

    # Save properties file
    output_file = output_dir / f'{domain_prefix}_reciter_config.properties'
    with open(output_file, 'w') as f:
        f.write(f'# ReCiter configuration for {institution_label}\n')
        f.write(f'# Generated by reciter_institution_setup.py\n')
        f.write(f'# Date: {time.strftime("%Y-%m-%d")}\n\n')
        for key, value in config.items():
            f.write(f'{key}={value}\n')
    print(f'\nSaved to: {output_file}')

    # Save JSON file (SPRING_APPLICATION_JSON format)
    spring_json = json.dumps(config, indent=2)
    json_file = output_dir / f'{domain_prefix}_spring_config.json'
    with open(json_file, 'w') as f:
        f.write(spring_json)
    print(f'JSON config: {json_file}')
    print(f'\nApply to your ReCiter instance:')
    compact_json = json.dumps(config)
    print(f'  # Option 1: Kubernetes (EKS)')
    print(f"  kubectl -n reciter set env deployment/reciter-dev SPRING_APPLICATION_JSON='{compact_json}'")
    print(f'  # Option 2: Docker Compose / local')
    print(f"  export SPRING_APPLICATION_JSON='{compact_json}'")


if __name__ == '__main__':
    main()
