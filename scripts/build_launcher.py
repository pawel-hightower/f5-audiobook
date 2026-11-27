"""Kompiluje natywny cichy launcher bAIarz.exe (15 KB, bez okna konsoli, z wbudowaną ikoną assets/baiarz.ico)
oraz opcjonalnie tworzy skróty w Menu Start i na Pulpicie Windows."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ICON_ICO = PROJECT_ROOT / "assets" / "baiarz.ico"
OUT_EXE = PROJECT_ROOT / "bAIarz.exe"
CSC_EXE = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")

LAUNCHER_CS = r"""
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        try
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');
            string pythonw = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
            string entryPyw = Path.Combine(baseDir, "bAIarz.pyw");

            if (!File.Exists(pythonw))
            {
                MessageBox.Show(
                    "Nie znaleziono środowiska Python:\n" + pythonw + "\n\nUpewnij się, że folder .venv istnieje.",
                    "bAIarz — błąd uruchomienia",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }
            if (!File.Exists(entryPyw))
            {
                MessageBox.Show(
                    "Nie znaleziono pliku startowego:\n" + entryPyw,
                    "bAIarz — błąd uruchomienia",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            var psi = new ProcessStartInfo
            {
                FileName = pythonw,
                Arguments = "\"" + entryPyw + "\"",
                WorkingDirectory = baseDir,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            psi.EnvironmentVariables["PYTHONPATH"] = Path.Combine(baseDir, "src");
            psi.EnvironmentVariables["QT_MEDIA_BACKEND"] = "windows";

            Process.Start(psi);
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "Nie udało się uruchomić aplikacji bAIarz:\n" + ex.Message,
                "bAIarz — błąd uruchomienia",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
    }
}
"""


def build_exe() -> Path:
    if not CSC_EXE.exists():
        raise FileNotFoundError(f"Brak kompilatora .NET: {CSC_EXE}")
    if not ICON_ICO.exists():
        raise FileNotFoundError(f"Brak pliku ikony: {ICON_ICO}")

    with tempfile.TemporaryDirectory() as td:
        cs_file = Path(td) / "bAIarzLauncher.cs"
        cs_file.write_text(LAUNCHER_CS, encoding="utf-8")
        cmd = [
            str(CSC_EXE),
            "/nologo",
            "/optimize+",
            "/target:winexe",
            f"/win32icon:{ICON_ICO}",
            f"/out:{OUT_EXE}",
            "/r:System.Windows.Forms.dll",
            str(cs_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Błąd kompilacji csc.exe:\n{res.stdout}\n{res.stderr}")
    print(f"Skompilowano launcher: {OUT_EXE} ({OUT_EXE.stat().st_size} B)")
    return OUT_EXE


def install_shortcuts() -> None:
    if not OUT_EXE.exists():
        raise FileNotFoundError(f"Najpierw skompiluj {OUT_EXE}")

    ps_script = f"""
$ws = New-Object -ComObject WScript.Shell
$targets = @(
    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'bAIarz.lnk'),
    (Join-Path ([Environment]::GetFolderPath('Programs')) 'bAIarz.lnk')
)
foreach ($lnkPath in $targets) {{
    $sc = $ws.CreateShortcut($lnkPath)
    $sc.TargetPath = '{OUT_EXE}'
    $sc.WorkingDirectory = '{PROJECT_ROOT}'
    $sc.IconLocation = '{ICON_ICO},0'
    $sc.Description = 'bAIarz — Generator Polskich Audiobooków AI'
    $sc.Save()
    Write-Output "Utworzono skrót: $lnkPath"
}}
"""
    res = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True,
        text=True,
        check=True,
    )
    print(res.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe-only", action="store_true", help="Tylko skompiluj bAIarz.exe")
    ap.add_argument("--shortcuts-only", action="store_true", help="Tylko utwórz skróty Windows")
    args = ap.parse_args()

    if not args.shortcuts_only:
        build_exe()
    if not args.exe_only:
        install_shortcuts()


if __name__ == "__main__":
    main()
