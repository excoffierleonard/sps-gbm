use assert_cmd::Command;
use predicates::prelude::*;

#[test]
#[ignore] // Requires a valid API key and network connection
fn test_cli_basic() {
    let mut cmd = Command::cargo_bin("sps-gbm").unwrap();

    let assert = cmd.assert();
    assert
        .success()
        .stdout(predicate::str::contains("AAPL"))
        .stdout(predicate::str::contains(".png"));
}

#[test]
#[ignore] // Requires a valid API key and network connection
fn test_cli_with_ticker() {
    let mut cmd = Command::cargo_bin("sps-gbm").unwrap();

    let assert = cmd.arg("-t").arg("AAPL").assert();
    assert
        .success()
        .stdout(predicate::str::contains("AAPL"))
        .stdout(predicate::str::contains(".png"));
}
