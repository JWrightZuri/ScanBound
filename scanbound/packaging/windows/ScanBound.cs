// ScanBound.exe: starts the ScanBound app with the Python runtime installed next to it.
//
//   ScanBound.exe [data folder or project.json]
//
// Runs  runtime\pythonw.exe -m scanbound.app <arguments>  without a console window and waits for it.
// If the app stops with an error, the last lines it printed are shown and saved to
// %LOCALAPPDATA%\ScanBound\last_crash.txt.  Compiled by build_installer.ps1 with the .NET Framework's csc.exe.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("ScanBound")]
[assembly: System.Reflection.AssemblyProduct("ScanBound")]

static class Launcher
{
    const int KeepLines = 40;
    static readonly Queue<string> tail = new Queue<string>();

    [STAThread]
    static int Main(string[] args)
    {
        string home = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(home, "runtime", "pythonw.exe");
        if (!File.Exists(python))
        {
            MessageBox.Show("ScanBound's Python runtime is missing:\n" + python + "\n\nPlease reinstall ScanBound.",
                            "ScanBound", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        var cmd = new StringBuilder("-m scanbound.app");
        foreach (string a in args) cmd.Append(' ').Append(Quote(a));
        var info = new ProcessStartInfo(python, cmd.ToString());
        info.UseShellExecute = false;
        info.RedirectStandardError = true;
        info.CreateNoWindow = true;
        // keep the bundled runtime to itself: no user site-packages or a PYTHONPATH meant for another Python
        info.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
        info.EnvironmentVariables.Remove("PYTHONPATH");
        info.EnvironmentVariables.Remove("PYTHONHOME");

        using (var p = new Process())
        {
            p.StartInfo = info;
            p.ErrorDataReceived += (s, e) => {
                if (e.Data == null) return;
                lock (tail) { tail.Enqueue(e.Data); while (tail.Count > KeepLines) tail.Dequeue(); }
            };
            try { p.Start(); }
            catch (Exception e)
            {
                MessageBox.Show("Could not start ScanBound:\n" + e.Message, "ScanBound",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
            p.BeginErrorReadLine();
            p.WaitForExit();
            if (p.ExitCode != 0) Report(p.ExitCode);
            return p.ExitCode;
        }
    }

    static void Report(int code)
    {
        string text;
        lock (tail) text = string.Join(Environment.NewLine, tail.ToArray());
        string saved = "";
        try
        {
            string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ScanBound");
            Directory.CreateDirectory(dir);
            string file = Path.Combine(dir, "last_crash.txt");
            File.WriteAllText(file, DateTime.Now + "  exit code " + code + Environment.NewLine + text);
            saved = "\n\nSaved to " + file;
        }
        catch (Exception) { }
        MessageBox.Show("ScanBound stopped unexpectedly (exit code " + code + ").\n\n" + text + saved,
                        "ScanBound", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    // quoting that CommandLineToArgvW (and so Python) splits back into the original arguments
    static string Quote(string a)
    {
        if (a.Length > 0 && a.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) return a;
        var sb = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in a)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') sb.Append('\\', slashes * 2 + 1);
            else sb.Append('\\', slashes);
            slashes = 0;
            sb.Append(c);
        }
        sb.Append('\\', slashes * 2).Append('"');
        return sb.ToString();
    }
}
