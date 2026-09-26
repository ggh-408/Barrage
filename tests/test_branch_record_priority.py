import pickle

from barrage_rl.branch_records import BranchRecorder


def test_pre_saved_snapshot_is_promoted_to_failure_tail_without_replacement(tmp_path):
    recorder=BranchRecorder(tmp_path,0)
    item=dict(key='env0_episode0_step90',payload=[1,2,3])
    recorder.save(item,'uniform_control')
    recorder.save({**item,'payload':[9]},'near_miss_candidate')
    recorder.tail.append(item)
    recorder.failed()
    recorder.save(item,'near_miss_candidate')
    recorder.save(item,'uniform_control')
    with (tmp_path/(item['key']+'.pkl')).open('rb') as f:
        stored=pickle.load(f)
    assert stored['reason']=='failure_tail'
    assert stored['payload']==[1,2,3]
    assert not list(tmp_path.glob('*.tmp'))


def test_failure_already_saved_before_flush_keeps_its_priority(tmp_path):
    recorder=BranchRecorder(tmp_path,0)
    item=dict(key='env0_episode0_step99')
    recorder.save(item,'failure_tail')
    recorder.near_candidates[0]=((5,7,0.),item)
    recorder.flush()
    with (tmp_path/(item['key']+'.pkl')).open('rb') as f:
        assert pickle.load(f)['reason']=='failure_tail'
