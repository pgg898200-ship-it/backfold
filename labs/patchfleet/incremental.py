"""Build a migrated Git tree, parsing each changed source revision once."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from build import FLAGS, recipe, reference_index
from fetch import git_cat
from fleet import git, run_recipe

def migrate(repo, revision, cache):
    if subprocess.check_output(["ruff","--version"]).strip() != b"ruff 0.11.13":
        raise ValueError("Use ruff==0.11.13")
    root=git(repo,"rev-parse","--verify","--end-of-options",revision+"^{tree}").decode().strip()
    tree=git(repo,"ls-tree","-rz",root)
    entries=[]
    for record in tree.rstrip(b"\0").split(b"\0"):
        meta,name=record.split(b"\t",1)
        mode,_,oid=meta.split()
        if mode in (b"100644",b"100755") and name.endswith(b".py"):
            entries.append((name.decode(),oid))
    # The table is scoped to this exact rule, tool version, options and staging contract.
    db=sqlite3.connect(cache,timeout=60)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS npy201_01113_py39_v1 (path TEXT, oid BLOB, edit TEXT, PRIMARY KEY(path,oid))")
        cold,hit=[],[]
        for name,oid in entries:
            row=db.execute("SELECT edit FROM npy201_01113_py39_v1 WHERE path=? AND oid=?",(name,oid)).fetchone()
            if row is None:
                cold.append((name,oid))
            elif row[0]:
                hit.append((name,oid,json.loads(row[0])))
        changed={}
        material=git_cat(repo,[(p,s) for p,s,_ in hit])
        for (name,data),(_,_,program) in zip(material,hit,strict=True):
            changed[name]=run_recipe(data,program)
        updates=[]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            raw=dict(git_cat(repo,cold))
            for name,data in raw.items():
                dest=directory/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(data)
            names=list(raw)
            for i in range(0,len(names),500):
                subprocess.run(["ruff",*FLAGS,"--",*names[i:i+500]],cwd=directory,check=True,stdout=subprocess.DEVNULL)
            for name,oid in cold:
                data=(directory/name).read_bytes()
                code=None
                if data!=raw[name]:
                    changed[name]=data
                    code=json.dumps(recipe(raw[name],data),separators=(",",":"))
                updates.append((name,oid,code))
        target=reference_index(repo,root,changed)
        with db:
            db.executemany("INSERT OR IGNORE INTO npy201_01113_py39_v1 VALUES (?,?,?)",updates)
        return dict(source_tree=root,target_tree=target,python_paths=len(entries),parsed=len(cold),changed=len(changed))
    finally:
        db.close()
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path);p.add_argument("revision");p.add_argument("cache",type=Path)
    a=p.parse_args()
    print(json.dumps(migrate(a.repo.resolve(),a.revision,a.cache),indent=2))
