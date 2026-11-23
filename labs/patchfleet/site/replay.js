const text = new TextDecoder();
const enc = new TextEncoder();
const from64 = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));
function join(blocks) {
  const out = new Uint8Array(blocks.reduce((n,b) => n+b.length,0));
  let pos = 0;
  for (const block of blocks) { out.set(block,pos); pos+=block.length; }
  return out;
}
function replay(bytes,[kind,ops]) {
  if (kind==="replace") {
    for (const [before,after] of ops) {
      const find=from64(before), put=from64(after), pieces=[];
      let start=0;
      for (let i=0;i+find.length<=bytes.length;) {
        if (find.every((b,j)=>bytes[i+j]===b)) {
          pieces.push(bytes.slice(start,i),put);
          i+=find.length; start=i;
        } else {i++;}
      }
      bytes=join([...pieces,bytes.slice(start)]);
    }
    return bytes;
  }
  let cursor=0;const parts=[];
  for (const [offset,length,addition] of ops) {
    parts.push(bytes.slice(cursor,offset),from64(addition));cursor=offset+length;
  }
  return join([...parts,bytes.slice(cursor)]);
}
async function oid(bytes) {
  const payload=join([enc.encode("blob "+bytes.length+"\0"),bytes]);
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-1",payload)),x=>x.toString(16).padStart(2,"0")).join("");
}
const select=document.querySelector("#cases"),run=document.querySelector("#run");
const status=document.querySelector("#status"),view=document.querySelector("#changes");
async function fetchOK(url){const r=await fetch(url); if(!r.ok)throw new Error("HTTP "+r.status); return r;}
try {
  const data=await (await fetchOK("cases.json")).json();
  for(const [i,c] of data.entries()) select.add(new Option(c.release,i));
  run.disabled=false;status.textContent="Choose a release and replay its edit.";
  select.onchange=()=>{view.hidden=true;status.textContent="Ready to replay.";};
  run.onclick=async()=>{
    run.disabled=true;select.disabled=true;view.hidden=true;status.textContent="Fetching upstream file…";
    try {
      const c=data[Number(select.value)],src=new Uint8Array(await (await fetchOK(c.source)).arrayBuffer());
      if(await oid(src)!==c.before)throw new Error("Original file failed verification");
      const out=replay(src,c.recipe),actual=await oid(out);
      if(actual!==c.after)throw new Error("Output failed verification");
      const old=text.decode(src).split("\n"),next=text.decode(out).split("\n");
      let i=0;while(i<old.length && old[i]===next[i])i++;
      view.textContent=c.path+"\n\n"+old.slice(i,i+3).map(s=>"- "+s).join("\n")+"\n"+next.slice(i,i+3).map(s=>"+ "+s).join("\n");
      view.hidden=false;status.textContent="Verified output · "+actual;
    } catch(e){status.textContent="Replay failed · "+e.message;}
    finally{run.disabled=false;select.disabled=false;}
  };
} catch(e){status.textContent="Could not load the release cases · "+e.message;}
