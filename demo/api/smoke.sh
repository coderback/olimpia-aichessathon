#!/usr/bin/env bash
# Exercises a running API: usage ./smoke.sh https://host [allowed-origin]
set -u
BASE="${1:-http://localhost:18080}"
ORIGIN="${2:-https://x.test}"
fail=0

check() { # name expected-status actual-status body
  if [ "$2" = "$3" ]; then echo "ok    $1 -> $3  ${4:0:150}"; else echo "FAIL  $1 expected $2 got $3  ${4:0:150}"; fail=1; fi
}
post() { # body
  curl -s -m 60 -o /tmp/smoke_body -w '%{http_code}' -X POST "$BASE/move" -H 'Content-Type: application/json' -d "$1"
}
run() { # name expected body
  local code; code=$(post "$3"); check "$1" "$2" "$code" "$(cat /tmp/smoke_body)"
}

run "start position"        200 '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
run "after 1.e4"            200 '{"fen":"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"}'
run "mate in one"           200 '{"fen":"6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"}'
run "promotion"             200 '{"fen":"8/P6k/8/8/8/8/8/K7 w - - 0 1"}'
run "not json"              422 'hello'
run "wrong shape"           422 '{"fen":"x","extra":1}'
run "fen not a string"      422 '{"fen":5}'
run "garbage fen"           422 '{"fen":"not a fen at all"}'
run "no kings"              422 '{"fen":"8/8/8/8/8/8/8/8 w - - 0 1"}'
run "trailing junk"         422 '{"fen":"k7/8/8/8/8/8/8/R3K2R w - - 0 1x"}'
run "game over (checkmate)" 422 '{"fen":"3R2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1"}'

big=$(head -c 2000 /dev/zero | tr '\0' 'a')
run "body too large"        413 "{\"fen\":\"$big\"}"

code=$(curl -s -m 10 -o /tmp/smoke_body -w '%{http_code}' -X POST "$BASE/move" -H 'Content-Type: text/plain' -d '{}')
check "wrong content type" 415 "$code" "$(cat /tmp/smoke_body)"
code=$(curl -s -m 10 -o /tmp/smoke_body -w '%{http_code}' "$BASE/nope")
check "unknown path" 404 "$code" "$(cat /tmp/smoke_body)"
code=$(curl -s -m 10 -o /tmp/smoke_body -w '%{http_code}' "$BASE/healthz")
check "health" 200 "$code" "$(cat /tmp/smoke_body)"

acao=$(curl -s -m 10 -D - -o /dev/null -H "Origin: $ORIGIN" "$BASE/healthz" | tr -d '\r' | grep -i '^access-control-allow-origin' | cut -d' ' -f2)
check "CORS allowed origin" "$ORIGIN" "$acao" ""
acao=$(curl -s -m 10 -D - -o /dev/null -H "Origin: https://evil.example" "$BASE/healthz" | tr -d '\r' | grep -ci '^access-control-allow-origin')
check "CORS other origin refused" "0" "$acao" ""

seen429=no
for _ in $(seq 1 60); do
  code=$(post 'hello')
  if [ "$code" = "429" ]; then seen429=yes; break; fi
done
check "rate limit reached" yes "$seen429" ""

rm -f /tmp/smoke_body
exit $fail
