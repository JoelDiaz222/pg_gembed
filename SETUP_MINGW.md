# Setting Up pg_gembed on Windows with MSYS2 & MinGW

This guide walks you through setting up the `pg_gembed` PostgreSQL extension on Windows using [MSYS2](https://www.msys2.org/) and the MinGW-w64 (UCRT64) toolchain.

---

## Prerequisites

Download and install **MSYS2** from the official website:

👉 [https://www.msys2.org/](https://www.msys2.org/)

After installation, open the **UCRT64** terminal (recommended) from the MSYS2 start menu entries.

---

## 1. Install Required Dependencies

Run the following commands inside the MSYS2 terminal to install all required packages:

```bash
# Git
pacman -S git

# PostgreSQL (UCRT64 build)
pacman -S mingw-w64-ucrt-x86_64-postgresql

# Make and pkg-config
pacman -S make mingw-w64-ucrt-x86_64-pkgconf

# Rust toolchain
pacman -S mingw-w64-ucrt-x86_64-rust

# CA certificates (needed for secure connections)
pacman -S ca-certificates
update-ca-trust
```

---

## 2. Update the PATH

Add the UCRT64 binaries to your PATH so tools like `pg_config`, `psql`, etc. are available:

```bash
export PATH=$PATH:/c/msys64/ucrt64/bin
```

> **Tip:** Add this line to your `~/.bashrc` or `~/.bash_profile` to make it permanent across sessions.

---

## 3. Initialize and Start PostgreSQL

```bash
# Initialize the database cluster
initdb -D /usr/local/var/postgres

# Start the PostgreSQL server (logs to "logfile")
pg_ctl -D /usr/local/var/postgres -l logfile start

# Create a database for your user (replace <username> with your username)
createdb <username>
```

---

## 4. Install the pgvector Extension

`pg_gembed` depends on [pgvector](https://github.com/pgvector/pgvector) for vector storage support.

```bash
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
make install
cd ..
```

---

## 5. Build and Install pg_gembed

```bash
# Clone the main extension repository
git clone https://github.com/JoelDiaz222/pg_gembed
cd pg_gembed

# Clone the Rust embedding library (gembed)
git clone https://github.com/JoelDiaz222/gembed
cd gembed

# Build the Rust library in release mode
# (SSL verification is disabled for corporate/self-signed certificate environments)
GIT_SSL_NO_VERIFY=true CARGO_NET_GIT_FETCH_WITH_CLI=true cargo build --release

cd ..

# Install the PostgreSQL extension
make install
```

---

## 6. Enable the Extensions in PostgreSQL

Connect to PostgreSQL and activate both extensions:

```bash
psql
```

Inside the `psql` prompt:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_gembed;
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pg_config` not found | Make sure `/c/msys64/ucrt64/bin` is in your `PATH` |
| SSL errors during `cargo build` | Use `GIT_SSL_NO_VERIFY=true` as shown in step 5 |
| `initdb` fails | Ensure no existing PostgreSQL data directory exists at the target path |
| `pacman` package not found | Run `pacman -Syu` first to update the package database |
