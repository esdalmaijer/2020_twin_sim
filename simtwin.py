import os
DIR = os.path.abspath(os.path.dirname(__file__))
import itertools

import numpy
from scipy.stats import pearsonr
from matplotlib import pyplot

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr("lavaan")
try:
    from rpy2.rinterface import RRuntimeError
except:
    from rpy2.rinterface_lib.embedded import RRuntimeError


def twin_sim(p_heritability, p_shared, p_nonshared, sim_var=True, \
    n_twins=1000, p_mz_twins=0.5, n_genes=10, alleles=range(-7,8), \
    p_alleles=None, env_m=0.0, env_sd=1.0, p_parent_env=0, \
    p_child_shared_env=0, p_child_nonshared_env=0, p_noise=0, noise_m=0.0, \
    noise_sd=1.0, do_sem=True):
    
    # If proportions indicate proportions of the variance, take their square
    # root so that the proportions match the simulated variance. After that,
    # Make sure that the proportions add up to 1, so that the SD of the total
    # trait distribution is 1.
    if sim_var:
        # Heritability and environment.
        p_heritability = numpy.sqrt(p_heritability)
        p_shared = numpy.sqrt(p_shared)
        p_nonshared = numpy.sqrt(p_nonshared)
        scale_factor = 1.0 / numpy.sum([p_heritability, p_shared, p_nonshared])
        p_heritability *= scale_factor
        p_shared *= scale_factor
        p_nonshared *= scale_factor
        # Indirect genetic effects on shared environment.
        p_non_confounded_shared = 1.0 - p_parent_env - p_child_shared_env
        p_non_confounded_shared = numpy.sqrt(p_non_confounded_shared)
        p_parent_env = numpy.sqrt(p_parent_env)
        p_child_shared_env = numpy.sqrt(p_child_shared_env)
        scale_factor = 1.0 / numpy.sum([p_non_confounded_shared, \
            p_parent_env, p_child_shared_env])
        p_non_confounded_shared *= scale_factor
        p_parent_env *= scale_factor
        p_child_shared_env *= scale_factor
        # Indirect genetic effects on non-shared environments.
        p_nonchild_env = 1.0 - p_child_nonshared_env
        p_child_nonshared_env = numpy.sqrt(p_child_nonshared_env)
        scale_factor = 1.0 / (p_child_nonshared_env+numpy.sqrt(p_nonchild_env))
        p_child_nonshared_env *= scale_factor
        # Proportion of data impaired by measurement noise.
        p_signal = 1.0 - p_noise
        p_noise = numpy.sqrt(p_noise)
        scale_factor = 1.0 / (p_noise+numpy.sqrt(p_signal))
        p_noise *= scale_factor
    
    # Compute the number of monozygotic and dizygotic twins.
    n_mz = int(round(n_twins * p_mz_twins))
    n_dz = n_twins - n_mz
    
    # Create a population of mothers and fathers with random genotypes.
    m = numpy.random.choice(alleles, size=(n_genes, 2, n_twins), \
        replace=True, p=p_alleles)
    f = numpy.random.choice(alleles, size=(n_genes, 2, n_twins), \
        replace=True, p=p_alleles)
    # Split parents into twin groups.
    m_mz = m[:,:,:n_mz]
    f_mz = f[:,:,:n_mz]
    m_dz = m[:,:,n_mz:]
    f_dz = f[:,:,n_mz:]
    # Compute parent phenotypes.
    if (p_parent_env > 0) or (p_child_shared_env > 0):
        m_mz_p = numpy.mean(numpy.mean(m_mz, axis=0), axis=0)
        f_mz_p = numpy.mean(numpy.mean(f_mz, axis=0), axis=0)
        m_dz_p = numpy.mean(numpy.mean(m_dz, axis=0), axis=0)
        f_dz_p = numpy.mean(numpy.mean(f_dz, axis=0), axis=0)
        p_mz_p = (m_mz_p + f_mz_p)/2.0
        p_mz_p = p_mz_p.reshape(p_mz_p.shape[0],1)
        p_dz_p = (m_dz_p + f_dz_p)/2.0
        p_dz_p = p_dz_p.reshape(p_dz_p.shape[0],1)
    
    # Create monozygotic twins.
    mz = numpy.zeros((n_genes, 2, n_mz, 2), dtype=int)
    shape = (n_genes, n_mz)
    # Randomly select which allele each mother is going to pass on.
    mm_allele = numpy.random.choice([False,True], size=shape, \
        replace=True)
    mf_allele = mm_allele==False
    fm_allele = numpy.random.choice([False,True], size=shape, \
        replace=True)
    ff_allele = fm_allele==False
    # Loop through both siblings.
    for i in range(mz.shape[3]):
        # Pass on mothers' genes.
        mz[:,0,:,i][mm_allele] = m_mz[:,0,:][mm_allele]
        mz[:,0,:,i][mf_allele] = m_mz[:,1,:][mf_allele]
        # Pass on fathers' genes.
        mz[:,1,:,i][fm_allele] = f_mz[:,0,:][fm_allele]
        mz[:,1,:,i][ff_allele] = f_mz[:,1,:][ff_allele]
    # Compute monozygotic phenotype.
    p_mz = numpy.mean(numpy.mean(mz, axis=0), axis=0)
    # Simulate shared environment.
    se = env_m + numpy.random.randn(n_mz) * env_sd
    se_mz = numpy.zeros((n_mz, 2), dtype=float)
    se_mz[:,0] = numpy.copy(se)
    se_mz[:,1] = numpy.copy(se)
    if (p_parent_env > 0) or (p_child_shared_env > 0):
        se_mz = (1.0-p_parent_env-p_child_shared_env)*se_mz \
            + p_parent_env*p_mz_p + p_child_shared_env*p_mz
    # Simulate non-shared environment.
    nse_mz = env_m + numpy.random.randn(n_mz,2) * env_sd
    if p_child_nonshared_env > 0:
        nse_mz = (1.0-p_child_nonshared_env)*nse_mz \
            + p_child_nonshared_env*p_mz

    # Create dizygotic twins.
    dz = numpy.zeros((n_genes, 2, n_dz, 2), dtype=int)
    shape = (n_genes, n_dz)
    # Loop through both siblings.
    for i in range(dz.shape[3]):
        # Randomly select which allele each mother is going to pass on.
        mm_allele = numpy.random.choice([False,True], size=shape, \
            replace=True)
        mf_allele = mm_allele==False
        fm_allele = numpy.random.choice([False,True], size=shape, \
            replace=True)
        ff_allele = fm_allele==False
        # Pass on mothers' genes.
        dz[:,0,:,i][mm_allele] = m_dz[:,0,:][mm_allele]
        dz[:,0,:,i][mf_allele] = m_dz[:,1,:][mf_allele]
        # Pass on fathers' genes.
        dz[:,1,:,i][fm_allele] = f_dz[:,0,:][fm_allele]
        dz[:,1,:,i][ff_allele] = f_dz[:,1,:][ff_allele]
    # Compute dizygotic phenotype.
    p_dz = numpy.mean(numpy.mean(dz, axis=0), axis=0)
    # Simulate shared environment.
    se = env_m + numpy.random.randn(n_dz) * env_sd
    se_dz = numpy.zeros((n_dz, 2), dtype=float)
    se_dz[:,0] = numpy.copy(se)
    se_dz[:,1] = numpy.copy(se)
    if (p_parent_env > 0) or (p_child_shared_env > 0):
        se_dz = (1.0-p_parent_env-p_child_shared_env)*se_dz \
            + p_parent_env*p_dz_p + \
            p_child_shared_env*p_dz
    # Simulate non-shared environment.
    nse_dz = env_m + numpy.random.randn(n_dz,2) * env_sd
    if p_child_nonshared_env > 0:
        nse_dz = (1.0-p_child_nonshared_env)*nse_dz \
            + p_child_nonshared_env*p_dz
    
    # Combine phenotype, shared environment, and non-shared environment
    # into trait values.
    trait_mz = p_heritability*p_mz + p_shared*se_mz + p_nonshared*nse_mz
    trait_dz = p_heritability*p_dz + p_shared*se_dz + p_nonshared*nse_dz
    # Apply normally distributed noise to simulate measurement error.
    if p_noise > 0:
        noise_mz = noise_m + numpy.random.randn(n_mz,2) * noise_sd
        trait_mz = (1.0-p_noise)*trait_mz + p_noise*noise_mz
        noise_dz = noise_m + numpy.random.randn(n_dz,2) * noise_sd
        trait_dz = (1.0-p_noise)*trait_dz + p_noise*noise_dz
    
    # Compute the estimates from the standard equations.
    r_mz, p_mz = pearsonr(trait_mz[:,0], trait_mz[:,1])
    r_dz, p_dz = pearsonr(trait_dz[:,0], trait_dz[:,1])
    tra_her = 2 * (r_mz-r_dz)
    tra_sha = (2*r_dz) - r_mz
    tra_non = 1.0 - r_mz
    
    # Return without doing a Lavaan estimate if we're skipping SEM.
    if not do_sem:
        result = { \
            "r_mz": r_mz, \
            "r_dz": r_dz, \
            "tra_her": tra_her, \
            "tra_sha": tra_sha, \
            "tra_non": tra_non, \
            "sem_her": numpy.NaN, \
            "sem_sha": numpy.NaN, \
            "sem_non": numpy.NaN, \
            "r_error": numpy.NaN, \
            }
        return result
    
    # Compute covariance matrices for MZ and DZ twins.
    cov_mz = numpy.cov(trait_mz.T)
    cov_mz_flat = cov_mz.reshape(4)
    cov_dz = numpy.cov(trait_dz.T)
    cov_dz_flat = cov_dz.reshape(4)

    # Create an R string to combine covariances.
    r_str = """function(){"""
    r_str += """
    MZY<-matrix(c({},{},{},{}),nrow=2)
    rownames(MZY)<-c("P1" , "P2")
    colnames(MZY)<-c("P1" , "P2")
    DZY<-matrix(c({},{},{},{}),nrow=2)
    rownames(DZY)<-c("P1" , "P2")
    colnames(DZY)<-c("P1" , "P2")
    """.format( \
        cov_mz_flat[0], cov_mz_flat[1], cov_mz_flat[2], cov_mz_flat[3], \
        cov_dz_flat[0], cov_dz_flat[1], cov_dz_flat[2], cov_dz_flat[3])
    # Add sample size to the covariance matrices.
    r_str += """
    data.cov<-list(MZ=MZY,DZ=DZY)
    data.n<-list(MZ={},DZ={})
    """.format(trait_mz.shape[0], trait_dz.shape[0])
    # Add ACE model in lavaan syntax.
    r_str += """
    data.ace.model<-'
    # genetic and shared environment model
    A1 =~ NA*P1 + c(a,a)*P1
    A2 =~ NA*P2 + c(a,a)*P2
    C1 =~ NA*P1 + c(c,c)*P1
    C2 =~ NA*P2 + c(c,c)*P2
    # variances
    A1 ~~ 1*A1
    A2 ~~ 1*A2
    C1 ~~ 1*C1
    C2 ~~ 1*C2
    P1 ~~ c(e_sq,e_sq)*P1
    P2 ~~ c(e_sq,e_sq)*P2
    # covariances
    A1 ~~ c(1.0,0.5)*A2
    A1 ~~ 0*C1 + 0*C2
    A2 ~~ 0*C1 + 0*C2
    C1 ~~ c(1,1)*C2
    '
    """
    # Add model fitting code
    r_str += """
    fit.m<-c("chisq", "df", "pvalue" , "aic", "rmsea", "srmr")
    data.ace.fit<-cfa(data.ace.model,sample.cov=data.cov,sample.nobs=data.n)
    #meas<-fitMeasures(data.ace.fit, fit.m)
    #summary(data.ace.fit, standardized=TRUE)
    param<-parameterEstimates(data.ace.fit, standardized=TRUE)
    param
    """
    r_str += "}"
    # This thing can throw a Lavaan error if the sample covariance matrix is 
    # not positive-definite.
    try:
        # Construct a function from the R string.
        rfunc = robjects.r(r_str)
        # Run the function. Parameter estimates will be returned.
        out = rfunc()
        #out.to_csvfile(os.path.join(DIR,"test.csv"))
        # Get the estimates from the std.all column, which gives a**2, c**2, 
        # and e. (a and c still have to be squared)
        sem_result = {}
        ci = out.colnames.index("std.all")
        for li, lbl in enumerate(list(out[out.colnames.index("label")])):
            if lbl != "":
                sem_result[lbl] = out[ci][li]
        # Compute heritability, shared environment, and non-shared environment
        # estimates.
        sem_her = sem_result["a"]**2
        sem_sha = sem_result["c"]**2
        sem_non = sem_result["e_sq"]
        r_error = 0
    
    # Catch any R errors.
    except RRuntimeError:
        sem_her = numpy.NaN
        sem_sha = numpy.NaN
        sem_non = numpy.NaN
        r_error = 1
        
    # Combine and return the results.
    result = { \
        "r_mz": r_mz, \
        "r_dz": r_dz, \
        "tra_her": tra_her, \
        "tra_sha": tra_sha, \
        "tra_non": tra_non, \
        "sem_her": sem_her, \
        "sem_sha": sem_sha, \
        "sem_non": sem_non, \
        "r_error": r_error, \
        }

    return result

