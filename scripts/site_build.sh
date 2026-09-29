#!/usr/bin/env bash
# Build the site the way GitHub Pages does (legacy build = github-pages gem, UTC) into a folder outside the repo.
# The repo's Gemfile is NOT what GitHub uses, so this keeps its own build environment in ~/.cache/hk-blog-build.
#
# Usage: scripts/site_build.sh <out_dir> [--future] [--drafts]
#   --future  include posts dated after today (to check scheduled posts)
#   --drafts  include _drafts/
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${HK_BLOG_BUILD_DIR:-$HOME/.cache/hk-blog-build}"
RUBY_BIN="/opt/homebrew/opt/ruby@3.3/bin"     # github-pages 232 needs Ruby < 4.0
OUT="${1:?usage: scripts/site_build.sh <out_dir> [--future] [--drafts]}"
shift

[ -x "$RUBY_BIN/ruby" ] || { echo "Ruby 3.3 not found. Install it with: brew install ruby@3.3"; exit 2; }
mkdir -p "$ENV_DIR"
if [ ! -f "$ENV_DIR/Gemfile" ]; then
  cat > "$ENV_DIR/Gemfile" <<'EOF'
source "https://rubygems.org"
gem "github-pages", ">= 232", group: :jekyll_plugins
gem "jekyll-include-cache", group: :jekyll_plugins
gem "webrick"
# stdlib gems no longer bundled with newer Rubies
gem "csv"; gem "base64"; gem "bigdecimal"; gem "logger"; gem "ostruct"; gem "mutex_m"; gem "drb"; gem "faraday-retry"
EOF
fi
# Without GitHub API credentials, jekyll-github-metadata guesses a wrong site URL; pin it.
printf 'url: https://zayunsna.github.io\nbaseurl: ""\n' > "$ENV_DIR/_local.yml"

export PATH="$RUBY_BIN:$PATH" BUNDLE_GEMFILE="$ENV_DIR/Gemfile" BUNDLE_PATH="$ENV_DIR/vendor" JEKYLL_ENV=production TZ=UTC
bundle check >/dev/null 2>&1 || bundle install --quiet
bundle exec jekyll build -s "$REPO" -d "$OUT" --config "$REPO/_config.yml,$ENV_DIR/_local.yml" "$@"
