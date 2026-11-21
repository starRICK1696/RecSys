import numpy as np
import scipy.sparse as sp
import pytest
from RecSys.kt2_models import ResearchAE, mixture_weights
from RecSys.kt2_experiments import detection, shorten
from RecSys.engine import evaluate

@pytest.mark.parametrize('family', ['gaussian','student','cauchy'])
def test_mixtures_separate_known_groups(family):
    rng = np.random.default_rng(1)
    errors = np.exp(np.r_[rng.normal(-5,.2,100),rng.normal(0,.2,100)])
    w, state = mixture_weights(errors, family)
    assert w[:100].mean() > w[100:].mean() + .5
    assert np.isfinite(w).all()
    assert np.isclose(sum(state['mixing']), 1)
    assert np.allclose(w, mixture_weights(errors, family)[0])


def test_target_excluded_genre_and_projection():
    m = sp.csr_matrix([[1.,1.,0.],[0.,1.,1.],[1.,0.,0.]])
    features = np.array([[1.,0.],[0.,1.],[1.,1.]])
    model = ResearchAE(rank=2, features=features, method='unit').fit(m)
    z = model._target_design(0)
    assert np.allclose(z[0,:2], model.projection_[1])
    assert np.allclose(z[0,2:-1], features[1])
    assert np.allclose(z[2,:-1], 0)
    rr, cc = m.nonzero()
    expected = np.array([model._target_design(j)[u]@model.decoder_[:,j] for u,j in zip(rr,cc)])
    assert np.allclose(model.pair_scores(rr,cc), expected)


def test_exact_ridge_and_excluded_loss():
    m = sp.csr_matrix([[1.,1.,0.],[0.,1.,1.],[1.,0.,1.],[0.,0.,1.]])
    omitted = sp.csr_matrix(([1.], ([1],[0])), shape=m.shape)
    model = ResearchAE(rank=2, method='unit', regularization=2).fit(m, excluded=omitted)
    z = model._target_design(0); w = np.full(4, .01); w[[0,2]]=1; w[1]=0
    expected = np.linalg.solve(z.T@(w[:,None]*z)+2*np.eye(3), z.T@(w*m[:,0].toarray().ravel()))
    assert np.allclose(model.decoder_[:,0], expected)


def test_crossfit_excludes_edges_from_features_and_targets(monkeypatch):
    m = sp.csr_matrix(np.ones((5,6))-np.eye(5,6))
    calls = []
    original = ResearchAE._solve
    def inspect(self, weights, excluded=None):
        if excluded is not None:
            assert self.matrix_.multiply(excluded).nnz == 0
            calls.append(excluded.nnz)
        return original(self,weights,excluded)
    monkeypatch.setattr(ResearchAE,'_solve',inspect)
    result = ResearchAE(rank=2,outer=1,crossfit=3).fit(m)
    assert sum(calls) == m.nnz
    assert np.isfinite(result.confidence_).all()


def test_detection_ties_and_perfect_order():
    labels = [0,0,1,1]
    assert detection(labels,[0,0,0,0]) == {'auroc': .5,'ap': .5,'prevalence': .5}
    assert detection(labels,[0,.1,.8,.9])['ap'] == 1
    assert detection(labels,[0,.1,.8,.9])['auroc'] == 1
    assert detection([0,0],[0,1])['auroc'] is None


def test_fixed_mask_noisy_zero():
    class Fixed:
        def score(self,users,interactions):
            return np.tile([10.,9.,1.],(len(users),1))
    train = sp.csr_matrix([[0.,0.,1.]])
    original = sp.csr_matrix([[1.,0.,1.]])
    test = sp.csr_matrix([[0.,1.,0.]])
    assert evaluate(Fixed(),train,test,k=1).recall == 0
    assert evaluate(Fixed(),train,test,k=1,seen=original).recall == 1


def test_short_histories_nested():
    m=sp.csr_matrix(np.ones((3,30)))
    small=shorten(m,5,42); large=shorten(m,10,42)
    assert np.all(np.diff(small.indptr)==5)
    assert small.multiply(large).nnz==small.nnz


def test_movielens_mapping_and_runner(tmp_path):
    from RecSys.kt2_experiments import load_movielens, run, parser
    source=tmp_path/'data'; source.mkdir()
    # Deliberately noncontiguous, unordered item IDs.
    ids=np.arange(30)*3+100
    (source/'ratings.dat').write_text('\n'.join(f'{u}::{i}::5::0' for offset,u in enumerate([7,2,19,30]) for i in np.roll(ids,offset*7)[:22][::-1]))
    (source/'movies.dat').write_text('\n'.join(f'{i}::Film::'+('Comedy' if j%2 else 'Drama') for j,i in enumerate(ids)))
    matrix, features, mapping=load_movielens(source)
    assert matrix.shape==(4,30)
    assert mapping['raw_item_ids']==ids.tolist()
    assert features[0,mapping['genres'].index('Drama')]==1
    args=parser().parse_args(['--data',str(source),'--output',str(tmp_path/'out'),
        '--rank','2','--seeds','42','--lambdas','1','--noise-levels','0','.1','--suites','robustness'])
    rows=run(args)
    assert len(rows)==28
    assert all(np.isfinite(r['ndcg']) for r in rows)

    original = {path.name: path.read_bytes() for path in args.output.iterdir()}
    assert {'manifest.json', 'metrics.csv', 'groups.csv', 'weights_unit_42.npz'} <= original.keys()
    args.seeds = [99]
    args.noise_levels = [0]
    with pytest.raises(FileExistsError, match='Choose a fresh directory'):
        run(args)
    assert {path.name: path.read_bytes() for path in args.output.iterdir()} == original
