from sage.structure.indexed_generators import IndexedGenerators
from sage.algebras.lie_algebras.lie_algebra import (LieAlgebraFromAssociative,
                                                    LieAlgebraWithGenerators)
from sage.algebras.lie_algebras.lie_algebra_element import LieAlgebraElement
from sage.categories.lie_algebras import (LieAlgebras)
from sage.categories.cartesian_product import cartesian_product

from sage.algebras.lie_algebras.heisenberg import(HeisenbergAlgebra, InfiniteHeisenbergAlgebra)

import itertools

from sage.rings.polynomial.infinite_polynomial_ring import(InfinitePolynomialRing_dense)

from sage.combinat.partition import Partition

import re
from sage.rings.polynomial.infinite_polynomial_element import InfinitePolynomial_dense

class HeisenbergAlgebraVSpace(IndexedGenerators, LieAlgebraWithGenerators):
    r"""
    An (infinite) Heisenberg Algebra moelled on a vector space

    EXAMPLES:

        sage: M = matrix([[1,6],[6,1]])
        sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
        sage: H = HeisenbergAlgebraVSpace(QQ, v)
        sage: H
        Heisenberg algebra over Ambient quadratic space of dimension 2 over Rational Field
        Inner product matrix:
        [1 6]
        [6 1]

    """
    def __init__(self,R,v, labels=None):
        r"""
        Implement infinite Heisenberg algebra based on a given vector space

        INPUT:

        - ``R`` -- Base ring

        - ``v`` -- underlying vector space

        - ``labels`` -- Labels for the basis of v

        EXAMPLES:

            sage: M = matrix([[1,6],[6,1]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: H = HeisenbergAlgebraVSpace(QQ, v)
            sage: H
            Heisenberg algebra over Ambient quadratic space of dimension 2 over Rational Field
            Inner product matrix:
            [1 6]
            [6 1]
        """
        #Label cannot start with a number
        self._v = v
        if labels == None:
            self._labels = tuple(['v' + str(i) for i in range(0,dim(v))])
        else:
            self._labels = labels
        for l in self._labels:
            if  l[0].isdigit():
                raise ValueError("label cannot start with digit")
        self._labelled_basis = {self._labels[i] : v.gen(i) for i in range(0,dim(v))}
        S = cartesian_product([PositiveIntegers(), ['p','q'], self._labels])
        cat = LieAlgebras(R).WithBasis()
        #        cat = LieAlgebras(R).Nilpotent().WithBasis()
        LieAlgebraWithGenerators.__init__(self,R, index_set = S, category=cat)
        IndexedGenerators.__init__(self, S, prefix = '', bracket=False,
                                    latex_bracket = False, string_quotes = False)

    def _repr_(self):
        return "Heisenberg algebra over {}".format(self._v._repr_())

    def p(self,i,k):
        r"""
        annihilation operator, annihilating i points lying on a cycle with cohomology class v_k
        """
        return self.element_class(self, {'p%i%s' %(i,self._labels[k]) : self.base_ring().one() })

    def q(self, i,k):
        r"""
        creation operator, creating i points lying on a cycle with cohomology class v_k
        """
        return self.element_class(self, {'q%i%s' %(i,self._labels[k]) : self.base_ring().one() })

    def c(self):
        return self.element_class(self, {'c' : self.base_ring().one()})

    def _repr_term(self, m):
        return m

    def _latex_term(self, m):
        (pq, i, n) = HeisenbergAlgebraVSpace._monomial_to_parts(m)
        if len(str(m)) == 1:
            return m
        if pq == 'p':
            e = ''
        else:
            e = '-'
        return "\\mathfrak{p}_{%s}(%s)"%(e + str(i), n)

    @staticmethod
    def _monomial_to_parts(m):
        s = re.match('([pq])([0-9]+)(.+$)', str(m)).groups()
        return((s[0], int(s[1]), s[2]))

    def bracket_on_basis(self, x, y):
        if y == 'c':
            return self.zero()
        elif x[0] == 'p' and y[0] == 'q' and x[1] == y[1]:
            v1 = self._labelled_basis[x[2:]]
            v2 = self._labelled_basis[y[2:]]
            i = Integer(x[1])
            return self.term('c', i*(-1)^(i -1 )*(v1.inner_product(v2)))
        else:
            return self.zero()

    def basis(self):
        S = cartesian_product([PositiveIntegers(), ['p','q'], self._labels])
        I = DisjointUnionEnumeratedSets([Set(['c']), S])
        basis_elt = lambda t : t if t == self.monomial('c') else self.monomial(t[1]+ str(t[0]) + t[2])
        return Family(I, basis_elt, name = "basis map")

    class Element(LieAlgebraElement):
        pass
    #    def _act_on_(self, x, is_left):
    #        if not is_left:
    #            return NotImplementedError
    #        elif isinstance(x, FockElement):
    #            return x.act(self)

    def lie_algebra_generators(self):
        return Family(self._indices, lambda t : self.monomial(t[1]+ str(t[0]) + t[2]), name = 'generator map')

# %% In[]
class FockModule(InfinitePolynomialRing_dense):
    """
    Implements the fock module of an infinite heisenberg algebra as an infinite polynomial ring

            EXAMPLES:

                sage: M = matrix([[0,-1],[-1,2]])
                sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
                sage: F = FockModule(QQ, v, labels=('f','s'))
                sage: F
                Fock module of central charge 1 over Ambient quadratic space of dimension 2 over Rational Field
                Inner product matrix:
                [ 0 -1]
                [-1  2]
    """

    def __init__(self,R, v, labels=None, central_charge =1):
        """
            Return a basis of the degree n subspace of self

            INPUT:

                - ``R`` -- Base ring

                - ``v`` -- underlying vector space

                - ``labels`` -- Labels for the basis of v

                - ``central_charge`` -- The number by which the central element
                  of the Heisenberg algebra acts on self

            EXAMPLES:

                sage: M = matrix([[1,0],[0,1]])
                sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
                sage: F = FockModule(QQ, v, labels=('f','s'))
                sage: F
                Fock module of central charge 1 over Ambient quadratic space of dimension 2 over Rational Field
                Inner product matrix:
                [1 0]
                [0 1]
        """
        self._central_charge = central_charge
        #super(FockModule, self).__init__()
        self._v = v
        if labels == None:
            self._labels = tuple(['v' + str(i) for i in range(0,dim(v))])
        else:
            self._labels = labels
        for l in self._labels:
            if  l[0].isdigit():
                raise ValueError("label cannot start with digit")
        self._labelled_basis = {self._labels[i] : v.gen(i) for i in range(0,dim(v))}
        InfinitePolynomialRing_dense.__init__(self, R, names = tuple(self._labels),order = 'lex' )

    def __repr__(self):
        return "Fock module of central charge {} over {}".format(self._central_charge, self._v._repr_())

    def central_charge(self):
        """
        Return the central charge of self
        """
        return self._central_charge

    def v_pair(self, v1,v2):
        return self._labelled_basis[v1].inner_product(self._labelled_basis[v2])

    def _labellings_of_partition(self, p):
        l = len(p)
        weights = itertools.product(*[self._labels]*l)
        D = self.gens_dict()
        # make sure that self has enough slots here
        term = lambda lp : product([D[v + '_' + str(k)] for (k,v) in lp])
        return set([term(zip(p,w)) for w in weights])

    def wt_basis(self, n):
        """
        Return a basis of the degree n subspace of self

        INPUT:

        - ``n`` -- integer; Which degree to return a basis of

        OUTPUT: A basis of the degree n elements of self as a set

        EXAMPLES:

            sage: M = matrix([[1,0],[0,1]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: F = FockModule(QQ, v, labels=('f','s'))
            sage: F.wt_basis(3)
            {s_1^3, s_2*s_1, s_3, f_1*s_1^2, f_1*s_2, f_1^2*s_1, f_1^3, f_2*s_1, f_2*f_1, f_3}

        NOTE::

            Even though degree 0 elements belong to the underlying infinite polynomial ring, they are not included in the fock module, so do not occur as a basis.

        """
        parts = iter(Partitions(n))
        labs = self._labels
        out = list(set.union(*[self._labellings_of_partition(p) for p in parts]))
        out.sort()
        return(out)

    @staticmethod
    def _mod_monomial_to_parts(m):
        s = re.match('^([a-zA-Z].*)_([0-9]+)$', str(m)).groups()
        return((s[0], int(s[1]), s[2]))

    def trace_at_wt(self, t, n):
        """
        Returns the trace of an operator on the degree n part of self

        INPUT:

        - ``t`` -- List [t1,t2,t3,..,tk] of elements of Heisenberg algebra
          represeneting the operator tk...t3t2t1 acting on self

        - ``n`` -- integer, the degree on which we intend to take the trace

        OUTPUT: A number, the trace of t on the degree n part of self

        EXAMPLES:

        This example shows that the trace of c is the
        dimension of the weight k space times the central charge  ::

            sage: M = matrix([[1,0],[0,1]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: F = FockModule(QQ, v, labels=('f','s'), central_charge = 12)
            sage: H = HeisenbergAlgebraVSpace(QQ, v, labels = ('f','s'))
            sage: n = len(F.wt_basis(4))
            sage: c = F.central_charge()
            sage: F.trace_at_wt([H.c()],4) == n * c
            True

        """

        basis = self.wt_basis(n)
        tr = 0
        for b in basis:
            tb = FockElement(self,b)
            for ti in t:
                tb = tb.act(ti)
            l = list(tb) #returns list of (a,v) for a_1v_1 + a_2v_2 + ...
            for (a,v) in l:
                if v == b:
                    tr += a
        return tr

class FockElement(InfinitePolynomial_dense):
    """
    Element of fock module, modelled as a polynomial in an infinite number of variables
    """
    def __init__(self, A, p):
        InfinitePolynomial_dense.__init__(self, A, p)

    @cached_method
    def _act_basis_on_monomial(self, t):
        R = self.parent()
        D = R.gens_dict()
        if t == t.parent().c():
            return (self.parent().central_charge())*self
        else:
            (pq, i, tv) = HeisenbergAlgebraVSpace._monomial_to_parts(t)
        if pq == 'q':
            key = tv+'_' + str(i)
            v = D[key]
            self.parent(v)
            #print "v = {}, v.parent() = {}".format(v, v.parent())
            #print "self = {}, self.parent() = {}".format(self,self.parent())
            return self*v
        elif pq == 'p': #recursive annihilate using
        #   p_i(v)q_i(v')^n = (-1)^(i-1) ni<v,v'>q_i^(n-1)(v') + q_i(v')^np_i(v)
            if self == 1:
                return self.parent().zero()
            labs = self.parent()._labels
            nlabs = len(labs)
            vs = self.variables()
            #find_ht = lambda n : int(re.search('[0-9]+$',  str(n)).group())
            exps = self.exponents()[0] #returns (a_ht, ..., a_1, b_ht, ..., b_1, ..., j_ht, ..., j_1) on v1^a v_2^b ... v_n^j
            ht = len(exps)/nlabs#need to do this because self.exponents() spits out a varying length tuple depending on which of the infinitely many variables occur in self
            for k in range(len(exps)):
                if exps[k] == 0:
                    continue
                else:
                    factor = lambda l : D[labs[floor(l/ht)] + '_' + str(Integer(ht-1 - (l%ht)))] # takes l and returns Fock variable v_i with v_i^exps[l] a factor of self
                    n = exps[k]
                    head = factor(k)
                    trailing = 1
                    for j in range(len(exps)):
                        if j == k:
                            continue
                        else:
                            trailing = factor(j)^(exps[j])*trailing
                    trailing = FockElement(self.parent(), trailing)
                    FockElement(self.parent(), head^n)
                    if i == ht-1 - (k%ht):
                        c = self.parent().central_charge()
                        #prefactor = (-1)^(i-1) *  i * c * n * self.parent().v_pair(tv, labs[floor(k/ht)])
                        prefactor =  i * c * n * self.parent().v_pair(tv, labs[floor(k/ht)])
                        add = prefactor*head^(n-1)*trailing
                    else:
                        add = 0
                    return(add + (head^n)*(trailing._act_basis_on_monomial(t)))
        else:
           raise ValueError('monomial in lie algebra is very wrong')

    @cached_method
    def act(self, t):
        """
        Returns the action of an element of the Heisenberg algebra on self.
        The p operators act by annihilation and the q operators act by creation

        INPUT:

        - ``t`` -- element of Heisenberg algebra on same vector space

        OUTPUT: The fock element returned when t acts on self

        EXAMPLES:

            illustrates some actions ::

            sage: M = matrix([[0,-1],[-1,2]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: F = FockModule(QQ, v, labels=('f','s'))
            sage: F.inject_variables()
            Defining f, s
            sage: a = FockElement(F, (8*f[6]^5))
            sage: L = HeisenbergAlgebraVSpace(QQ, v,labels = ('f','s'))
            sage: x = L.q(2,1)
            sage: y = L.p(7,1)
            sage: z = L.p(6,1)
            sage: c = L.c()
            sage: a.act(x)
            8*f_6^5*s_2
            sage: a.act(y)
            0
            sage: a.act(z)
            -240*f_6^4
            sage: a.act(c)
            8*f_6^5

            We can also use * multiplication from the left ::

            sage: x*a
            8*f_6^5*s_2

        """
        F = self.parent()
        ts = t.monomials()
        if self == 0:
            return FockElement(F, 0)
        ss = self.monomials()
        out = 0
        c_ts = lambda ti, sj : self.polynomial().monomial_coefficient(sj) * (t.monomial_coefficients()[str(ti)])
        return(sum([c_ts(ti, sj) * FockElement(F,sj)._act_basis_on_monomial(ti) for ti in ts for sj in ss]))

    @cached_method
    def act_annihilation_polynomial(self, t):
        """
        For t a polynomial of the right variable, returns the action of t on
        self where the polynomial is interpreted as an element of the universal
        enveloping algebra consisting entirely of annihilation operators.
        The creation operators case is easy, since it's simply multiplication.
        Then names must be STRING_123 for STRING names in FockModule


        INPUT:

        - ``t`` -- polynomial to be interpreted as annihilation operator

        OUTPUT: the action of the annihilaiton operator on self

        EXAMPLES:

        This example illustrates how to set up the appropriate polynomial ring ::

            sage: M = matrix([[1,0],[0,1]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: L = HeisenbergAlgebraVSpace(QQ, v, labels=('f','s'))
            sage: F = FockModule(QQ, v, labels=('f','s'), central_charge=1)
            sage: F.inject_variables()
            Defining f,s
            sage: x = FockElement(F,s[2]^10*f[1]^8)
            sage: R = PolynomialRing(QQ, 10, var_array = ['f_', 's_'])
            sage: R.inject_variables()
            Defining f_0, s_0, f_1, s_1, f_2, s_2, f_3, s_3, f_4, s_4, f_5, s_5, f_6, s_6, f_7, s_7, f_8, s_8, f_9, s_9
            sage: x.act(L.p(2,1))
            20*f_1^8*s_2^9
            sage: x.act_annihilation_polynomial(3*s_2^4*f_1^3)
            81285120*f_1^5*s_2^6
        """
        r = t.parent()
        vnames = r.variable_names()
        nvars = len(vnames)
        D = self.parent().gens_dict()
        L = HeisenbergAlgebraVSpace(QQ, self.parent()._v, self.parent()._labels)
        def monomial_factors(m):
            exps = list(m.exponents()[0])
            return ([(vnames[i], exps[i]) for i in range(nvars) if exps[i] != 0])
        label_index = lambda v : self.parent()._labels.index( re.search("[A-Za-z]+", v).group())
        lvl = lambda v : Integer(re.search("[0-9]+", v).group())
        m_coeff_pairs = [( t.monomial_coefficient(m), m) for m in t.monomials()]
        out = 0
        for (c, m) in m_coeff_pairs:
            fs = monomial_factors(m)
            summand = self
            for (vlabel, exp) in fs:
                idx = label_index(vlabel)
                n = lvl(vlabel)
                for j in range(exp):
                    summand = FockElement(self.parent(), summand.act(L.p(n , idx)))
            out = out +  c*summand
        return out


