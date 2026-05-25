from bubbleid_agent.cli import main


def test_cli_help_exits_successfully(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert "check-project" in output
    assert "write-report" in output
