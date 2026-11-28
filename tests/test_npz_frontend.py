"""NPZ imports use server conversion and retain real source provenance."""
from test_visual_frontend import run_app_js


def test_local_npz_import_uses_bounded_server_conversion():
    run_app_js("""
      context.validateReplay=data=>data;
      context.file={name:'local-policy.npz',size:32,text:async()=>{throw new Error('NPZ cannot be parsed as JSON');}};
      context.fetch=async(url,options)=>{
        assert.equal(url,'/api/replays/convert');assert.equal(options.method,'POST');assert.equal(options.body,context.file);
        return {ok:true,headers:{get:key=>key==='X-Replay-SHA256'?'real-file-hash':null},json:async()=>({task:'coverage',actor:'policy',scene:{},frames:[{t:0}]})};
      };
      await execute("document.getElementById('replay-file').listeners.change({target:{files:[file]}})");
      assert.equal(execute('state.replay.metadata.replay_sha256'),'real-file-hash');
      assert.equal(execute('state.replay.metadata.replay_path'),'local-policy.npz');
      context.file.size=128*1024*1024+1;
      context.fetch=()=>{throw new Error('Oversize import must be rejected before upload');};
      await execute("document.getElementById('replay-file').listeners.change({target:{files:[file]}})");
      assert(elements.get('status').textContent.includes('128'));
    """)


def test_http_npz_source_hash_is_preserved_for_manual_recording():
    run_app_js("""
      context.validateReplay=data=>data;
      context.fetch=async()=>({ok:true,headers:{get:()=> 'archive-sha256'},json:async()=>({task:'coverage',actor:'policy',scene:{},frames:[{t:0}]})});
      await execute("loadReplay('/outputs/eval/policy.npz')");
      assert.equal(execute('state.replay.metadata.replay_sha256'),'archive-sha256');
      assert.equal(execute('state.replay.metadata.replay_path'),'/outputs/eval/policy.npz');
    """)
