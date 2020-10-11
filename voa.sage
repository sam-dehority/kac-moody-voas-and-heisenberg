load ('Nakajima,Heisenberg.sage')
load ("PetersonMultilicity.sage")
# %% In[]
class LatticeVOA(SageObject):
    """Given an even lattice, methods for the Frenekl-Kac construction giving a representation of a BKM algebra on the fock space"""
    def __init__(self, R, v, labels =None):
        self._base_ring = R
        self._v = v
        self._fock = FockModule(R, v, labels)
        self._root_system = LorentzianBKMRootSystem(v.inner_product_matrix())
        self._heis = HeisenbergAlgebraVSpace(R, v, labels)
        self._rk = self._root_system._rk
        self._labels = self._fock._labels
        self._geo_pair = lambda a,b : -self._root_system.pair(a,b)
        self._lie_pair = self._root_system.pair

    def vec(self, a):
        ξ = self._v(a)
        ξ.set_immutable()
        return(ξ)

    def _repr_(self):
        return("Even Lattice Vertex Operator Algebra based on the " + repr(self._root_system))

    def __hash__(self):
        return(hash(repr(self)))

    def FK_tgt_wt(self, e_or_f, i, k, wt):
        alpha = self.vec([1 if j == i else 0 for j in range(self._rk)]) if e_or_f == 'e' else self.vec([-1 if j == i else 0 for j in range(self._rk)])
        return (wt[0] + alpha, wt[1] + k)

    def fourier_coeff_simple_on_wt_space(self, e_or_f, i, k, wt):
        r"""
        Returns a pair (target wt, [(create, annihilate)]) which captures the
        action of a fourier coefficient of a vertex operator on a given weight
        space of the Fock module. Only works for the fourier coefficients related to e_alpha(k) or f_alpha(k) where alpha is a simple root of the underlying BKM algebra.  For a general element, we must write it as a
        1) UEA element in the UEA of \hat{\mathfrak{g}}
        2) compute the composition of each term


        INPUT:

        - ``e_or_f`` -- character; either 'e' or 'f', and gives the corresponding
        generator

        - ``i`` -- integer; the index of which simple root. I.e. e_1(10) has i=1

        - ``k`` -- integer; the level of the operator, corresponds to e.g.
        e_i\otimes t^k = e_i(k) in the affinized algebra

        - ``wt`` -- tuple; describes the weight on which the fourier
        coefficient is meant to act

        OUTPUT: (target_wt, [creation_operator, anihilation_operator]) the second is a list [a_i, b_i] such that acton is the sum of a_i b_i.

        EXAMPLES:

        This example illustrates [e,f] = h on the weight space ((1,), -4)  of the basic representation of \hat{sl}_2, where h acts by a scalar 2. This space is the space of length 3 quotients of the line bundle with divisor the exceptional curve C in the A1 surface, i.e. with divisor P in T*P1.  It is therefore isomorphic to the cohomology of the  hilbert scheme A1^[3] ::

        sage: M = matrix([2])
        sage: v = VectorSpace(QQ,1,inner_product_matrix=M)
        sage: V = LatticeVOA(QQ, v, labels = ('c'))
        sage: e3 = V.frenkel_kac_simple_matrix('e', 0, 0, (V.vec((1,)),-4))
        sage: f3 = V.frenkel_kac_simple_matrix('f', 0,0, ( V.vec((2,)), -4))
        sage: f3p = V.frenkel_kac_simple_matrix('f', 0,0, ( V.vec((1,)), -4))
        sage: e3p = V.frenkel_kac_simple_matrix('e', 0,0, ( V.vec((0,)), -4))
        sage: e3p* f3p - f3*e3 # indirect doctest
        [2 0 0]
        [0 2 0]
        [0 0 2]


        """
        alpha = self.vec([1 if j == i else 0 for j in range(self._rk)]) if e_or_f == 'e' else self.vec([-1 if j == i else 0 for j in range(self._rk)])
        return self.fourier_coeff_on_wt_space(alpha, k, wt)

    def fourier_coeff_on_wt_space(self, α, k, wt):
        r"""
        Returns the fourier coefficient of the vertex operator X(α)

        INPUT:

        - ``α``  -- vector in v; which root to take VO of

        - ``k``  -- which fourier coefficient

        - ``wt``  -- source weight

        OUTPUT: (tgt_wt, [C,A]) just like for simples

        EXAMPLES:

        This example illustrates a fourier coefficient which is not a simple root ::

            sage: M = matrix([[2,-1],[-1,2]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('c', 'd'))
            sage: wt = (V.vec((0,0)), -3)
            sage: V.fourier_coeff_on_wt_space(V.vec((-1,-1)), 0, wt)
            (((-1, -1), -3),
            [(1, c_1 + d_1),
            (-c_1 - d_1, 1/2*c_1^2 + c_1*d_1 + 1/2*d_1^2 + 1/2*c_2 + 1/2*d_2),
            (-1/2*c_2 + 1/2*c_1^2 + c_1*d_1 - 1/2*d_2 + 1/2*d_1^2,
            1/6*c_1^3 + 1/2*c_1^2*d_1 + 1/2*c_1*d_1^2 + 1/6*d_1^3 + 1/2*c_1*c_2 + 1/2*d_1*c_2 + 1/2*c_1*d_2 + 1/2*d_1*d_2 + 1/3*c_3 + 1/3*d_3)])

        """
        n_pts = lambda c1, ch2 : Integer( self._geo_pair(c1, c1)/2 - ch2 )# root system , not NS pairing
        c1 = wt[0]
        ch2 = wt[1]
        n = n_pts(c1, ch2)
        source_n_pts = n
        uvects = [self.vec(list(1 if j == i else 0 for j in range(self._rk))) for i in range(self._rk)]
        summands = [self.vec(list(α[i] if j == i else 0 for j in range(self._rk)))  for m in range(self._rk)]
        neg_if_negs = [-1 if sign(c) < 0 else 0 for c in α] # i.e. (-1)^neg_if_f = -1 if f, else 1
        lbls = self._labels
        alpha_on_wt = -self._root_system.pair(c1, α)
        pair = self._root_system.pair

        #find target weight
        tgt_wt = (c1 + α, ch2 + k)
        tgt_c1 = tgt_wt[0]
        tgt_ch2 = tgt_wt[1]
        tgt_n_pts = n_pts(tgt_wt[0], tgt_wt[1])
        if n < 0:
            return (tgt_wt, [])

        #find cocycle
        #p = pair(c1, tgt_c1) + pair(c1, c1)*pair(tgt_c1, tgt_c1)
        #p = 2
        def cocycle_individual(c1p, c, i):
            c1_trunc = self.vec(tuple(c1[m] if m < i else 0 for m in range(self._rk)))
            p = self._lie_pair(c1_trunc, c*uvects[i])
            cocyc = (-1)^p
            return cocyc
        cocycle = 1
        c1c = c1
        for j in range(1, self._rk+1):
            cocycle *= cocycle_individual(c1c, α[self._rk - j], self._rk - j)
            c1c = c1c + summands[self._rk - j]

        # find annihilataion part of vertex operator
        highest_annihilation = 1 + n
        annihilation_coefficients = PolynomialRing(self._base_ring, highest_annihilation+1, var_array = [lbl +  '_' for lbl in self._labels ])
        annihilation_variables = annihilation_coefficients.gens()
        ann = {(lbl,j) : annihilation_variables[self._labels.index(lbl) + self._rk * j] for lbl in self._labels for j in range(1, highest_annihilation+1)} # v_0 are not allowed!
        R.<at> = PolynomialRing(annihilation_coefficients)
        AnnPolynomials.<a> = R.quotient(at^(highest_annihilation+1)) # a = z^-1 in vertex operator
        exp_ann = lambda y : sum(1/factorial(k) * y^k for k in
              range(highest_annihilation+1))
        α_ = lambda k : sum(α[j]*ann[(lbls[j], k)]  for j in range(self._rk))
        ann_factor = prod(exp_ann(- 1/k * α_(k) * a^k )
                            for k in range(1, highest_annihilation+1))
        ann_factor_coefficients = ann_factor.lift().dict()
        A = {-i : ann_factor_coefficients[i] for i in ann_factor_coefficients}

        #find creation part of vertex operator
        CrePolynomialsLift.<zl> = PolynomialRing(self._fock)
        D = self._fock.gens_dict()
        cre = lambda lbl, j : D[lbl  + '_' + str(j)]
        highest_create = Integer(tgt_n_pts)
        if highest_create < 0:
            return (tgt_wt, [])
        CrePolynomials.<z> = CrePolynomialsLift.quotient(zl^(highest_create+1))
        exp_cre = lambda y : sum(1/factorial(k) * y^k for k in
              range(highest_create+1))
        αminus = lambda k : sum(α[j]*cre(lbls[j], k)  for j in range(self._rk))
        cre_factor = prod(exp_cre( 1/k * αminus(k) * z^k)
                            for k in range(1, highest_create+1))
        cre_factor_coefficients = {0:1} if highest_create == 0 else cre_factor.lift().dict()
        C = cre_factor_coefficients

        # The z^k shifted coefficient x_k(alpha) of
        # X(alpha, z) = Cre(z)*e^a * z^shift * Ann(z^-1)
        #             = cocycle * Cre(z)* z^shift * A(a)
        # is
        #     cocycle * sum(C[j]*A[l] if j + l = N)
        N = tgt_n_pts - source_n_pts
        summands = [(cocycle * C[j], A[N - j])
                            for j in range(highest_create + 1)
                            if  N - j in A]
        return (tgt_wt, summands)

    def cre_ann_act_fock(self, L, x):
        r"""
        Retunrs the action of a list L of creation/annihilation pairs on a fock
        element x

        INPUT:

        - ``L`` -- list of tuples (create, annihilate)

        - ``x`` -- element of fock space

        OUTPUT: sum of create_i ann_i x
        """
        out = 0
        for (c,a) in L:
            afterann = FockElement(self._fock, x).act_annihilation_polynomial(a)
            o = InfinitePolynomial_dense(self._fock, c)*(FockElement(self._fock, x).act_annihilation_polynomial(a))
        return sum(InfinitePolynomial_dense(self._fock, c)*(FockElement(self._fock, x).act_annihilation_polynomial(a)) for (c, a) in L )

    def _n_pts(self, wt):
        return(Integer(self._geo_pair(wt[0], wt[0])/2 - wt[1]))

    def _root_space_casimir_summands_at_wt(self, root, k, wt, F_as_Lyndon = False):
        r"""
        Returns a list of matrices representing e_-α(-k)e_α(k) as α ranges over
        the basis for the root space root, and the -α form a dual basis

        INPUT:

        - ``root`` -- tuple; which root space. Positive roots frequently involve negative roots in the underlying non-affinized algebra.

        - ``k`` -- nonnegative integer; which level. If positive, root can by positive or negative. If zero, root must be positive

        - ``wt`` -- weight; which is the base weight on which we're acting.

        - ``F_as_Lyndon`` -- bool; If True, we return on the choice of dual bases where f_I are
          selected to be standard bracketed Lyndon words in the free Lie algebra on the f_i
          generators.

        OUTPUT: List of square matrices of dimension dim(wt)xdim(wt) of length mult(α)

        EXAMPLES:

        This example illustrates the utility of this package. Without
        nilpotency of some operators, we cannot use the usual way of caluclating these summands, or the traces would be equal. This is especially true in this instance, where the fourier coefficients do NOT form an affine/affinized Lie algebra ::

            sage: M = matrix([[2, -2,0],[-2,2,-1], [0,-1,2]])
            sage: v = VectorSpace(QQ,3,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('a', 'b', 'c'))
            sage: wt = (V.vec((0,1,0)), -2)
            sage: V._root_space_casimir_summands_at_wt(V.vec((2,2,1)), 2, wt)
            [
            [ -3  56 -16]  [  4   8  -4]
            [-18 109 -32]  [  2  22 -10]
            [-18 116 -31], [  3  24 -10]
            ]

        Now we demonstrate that this calculates the same thing as using
        the argument in the proof of Freudenthal's formula, which is calculated
        with the method ``casimir_summand_tr`` in the class
        ``BasicRepresenetation`` where it works. This is also an indirect test of
        the entire VOA code of the Frenkel Kac construction. ::

            sage: M = matrix([[2,-1],[-1,2]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('c', 'd'))
            sage: wt = (V.vec((0,0)), -4)
            sage: B = BasicRepresentation(V._root_system)
            sage: b = B.casimir_summand_tr(wt, (0,1),1);b
            15
            sage: a = V._root_space_casimir_summands_at_wt((0,1),1,wt)[0].trace();a
            15
            sage: a == b
            True

        """
        r = self.vec(root)
        root_positive = (sum(r) > 0)
        if k < 0 or (k == 0 and not root_positive):
            raise ValueError("Root must be positive")
        es, fs = self._root_system.root_space_basis(root,dual=True, monomials_in_F = F_as_Lyndon) if root_positive else self._root_system.root_space_basis(-r, dual= True, monomials_in_F = F_as_Lyndon)
        inter_wt = (wt[0] + r, wt[1]+k)
        there = lambda x : self.kac_moody_element_matrix(x, k, wt)
        back = lambda y : self.kac_moody_element_matrix(y, -k, inter_wt)
        if root_positive:
            return([back(fs[i]) * there(self._root_system._LyndonE(es[i])) for i in range(len(es))])
        else:
            return([back(self._root_system._LyndonE(es[i])) * there(fs[i]) for i in range(len(es))])

    @cached_method
    def frenkel_kac_simple_matrix(self, e_or_f, i, k, wt):
        r"""
        Returns a matrix which gives the action of e_i(k) or f_i(k) on
        wt space, with target the target wt space..shift of fourier coeff. by 1

        INPUT:

        - ``e_or_f, i, k, wt`` -- as in frenkel_kac_simple_on_wt_space

        OUTPUT: the rep of R e_i(k) or f_i(k) in gl(Fock) but only the block in Mat(Fock_wt, Fock_tgt_wt).

        EXAMPLES:

        This example illustrates [e,f] = h on the weight space ((1,), -4)  of the basic representation of \hat{sl}_2, where h acts by a scalar 2. This space is the cohomology of the space of length 3 quotients of the line bundle with divisor the exceptional curve C in the A1 surface, i.e. with divisor P1 in T*P1.  It is therefore isomorphic to the cohomology of the  hilbert scheme A1^[3] ::

            sage: M = matrix([2])
            sage: v = VectorSpace(QQ,1,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('c'))
            sage: e3 = V.frenkel_kac_simple_matrix('e', 0, 0, (V.vec((1,)),-4))
            sage: f3 = V.frenkel_kac_simple_matrix('f', 0,0, ( V.vec((2,)), -4))
            sage: f3p = V.frenkel_kac_simple_matrix('f', 0,0, ( V.vec((1,)), -4))
            sage: e3p = V.frenkel_kac_simple_matrix('e', 0,0, ( V.vec((0,)), -4))
            sage: e3p* f3p - f3*e3
            [2 0 0]
            [0 2 0]
            [0 0 2]

        """
        alpha = self.vec(vector(1 if j == i else 0 for j in range(self._rk))) if e_or_f == 'e' else self.vec(vector(-1 if j == i else 0 for j in range(self._rk)))
        return self.fourier_coefficient_matrix(alpha, k, wt)

    @cached_method
    def fourier_coefficient_matrix(self, α, k, wt):
        r"""
        Returns the marix of the fourier coefficient at a weight space

        INPUT:

        - ``α``  -- vector; which root to take VO of

        - ``k``  -- which fourier coefficient

        - ``wt``  -- source weight

        OUTPUT: (tgt_wt, [C,A]) just like for simples

        EXAMPLES:

        This example checks a commutator of roots which are not simple ::

            sage: M = matrix([[2,-1],[-1,2]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('c', 'd'))
            sage: f = V.fourier_coefficient_matrix(V.vec((-1,-1)), 0,(V.vec((1,0)), -3))
            sage: e = V.fourier_coefficient_matrix(V.vec((1,1)), 0, (V.vec((0,-1)),-3))
            sage: ep = V.fourier_coefficient_matrix(V.vec((1,1)), 0, (V.vec((1,0)),-3))
            sage: fp = V.fourier_coefficient_matrix(V.vec((-1,-1)), 0, (V.vec((2,1)),-3))
            sage: e*f - fp*ep
            [-1  0  0  0  0]
            [ 0 -1  0  0  0]
            [ 0  0 -1  0  0]
            [ 0  0  0 -1  0]
            [ 0  0  0  0 -1]

        """
        assert α.is_immutable(), f'vector {α} of type {type(α)} not hashable!!'
        assert (isinstance(wt[0], tuple) or self._v(wt[0]).is_immutable()), f"wt {wt} not immutable!"
        n_pts = lambda c1, ch2 :  self._geo_pair(c1, c1)/2 - ch2 # root system , not NS pairing
        (tgt_wt,L) = self.fourier_coeff_on_wt_space(α, k, wt)
        source_pts = Integer(n_pts(wt[0], wt[1]))
        tgt_pts = Integer(n_pts(tgt_wt[0], tgt_wt[1]))
        if tgt_pts < 0 or source_pts < 0:
            return 0
        source_basis = list(self._fock.wt_basis(source_pts))
        tgt_basis = list(self._fock.wt_basis(tgt_pts))
        tgt_dim = len(tgt_basis)
        source_dim = len(source_basis)
        R = Matrix(QQ, tgt_dim, source_dim)
        for i in range(source_dim):
            image_of_bi = self.cre_ann_act_fock(L, source_basis[i])
            if image_of_bi == 0:
                for j in range(tgt_dim):
                    R[j,i] = 0
            else:
                dctim = dict(image_of_bi)
                dct = {b : a for (a,b) in list(image_of_bi)}
                ci = [0]*tgt_dim
                for j in range(tgt_dim):
                    t_j = tgt_basis[j]
                    R[j,i] = dct[tgt_basis[j]] if t_j in dct else 0
                    ci[j] = R[j,i]
        return R

    def frenkel_kac_target(self, x, k, wt):
        r"""
        Returns the target wt for x acting on wt x, where x is in
        Lyndon basis for e_is or f_is.

        INPUT:

        - ``x`` -- element in _LyndonE or F.

        - ``wt`` -- weight acting on

        OUTPUT: The target wt
        """
        try:
            alpha = self._root_system.lyndon_root_space(x)
        except AttributeError:
            alpha = self._root_system.lyndon_root_space(x.list()[0][0])
        return((wt[0] + alpha, wt[1]+k))

    #def casimir_summand(self, x, k, x_opp, wt):
    def _commutator_at_wt(self, x1, k1, x2, k2, wt):
        #used to find the action of elements not in simple root spaces.
        #returs the commutator of x1 and x2 at the weight wt.
        # x1 and x2 don't both have to be the same f or es, which makes
        # this slightly different from the frenkel_kac_matrix method
        FK_tgt = self.frenkel_kac_target
        x1_tgt_from_wt = FK_tgt(x1, k1, wt)
        x2_tgt_from_wt = FK_tgt(x2, k2 , wt)
        final = FK_tgt(x1, k1, x2_tgt_from_wt)
        x1_from_wt = self.frenkel_kac_matrix(x1, k1, wt)
        x2_from_wt = self.frenkel_kac_matrix(x2, k2, wt)
        x1_to_final = self.frenkel_kac_matrix(x1, k1, x2_tgt_from_wt)
        x2_to_final = self.frenkel_kac_matrix(x2, k2, x1_tgt_from_wt)
        s1 = x1_to_final * x2_from_wt
        s2 = x2_to_final * x1_from_wt
        if s1 != 0 and s2 != 0:
            return (s1 - s2)
        elif s1 == 0:
            return (- s2)
        else:
            return s1

    def _test_commutator_at_wts_if_bdd(self, x1, k1, x2, k2, wts, highest = 6):
        for wt in wts:
            FK_tgt = self.frenkel_kac_target
            w2 = FK_tgt(x1, k1, wt)
            w3 = FK_tgt(x2, k2 , wt)
            w4 = FK_tgt(x1, k1, w3)
            n1, n2, n3, n4 = self._n_pts(wt), self._n_pts(w2), self._n_pts(w3), self._n_pts(w4)
            if n1 > highest or n2 > highest or n3 > highest or n4 > highest:
                print(f"wt {wt} has target w/ too many points")
            else:
                o = self._commutator_at_wt(x1, k1, x2, k2, wt)
                print(f"[{x1}({k1}),{x2}({k2})] on wt {wt} = \n{o}")

    def frenkel_kac_matrix(self, x,k, wt, lvl_in_right = True):
        r"""
        Returns the action of a monomial of the free lie algebra x
        on the wt space at level k, i.e. x t^k in the affinized.

        INPUT:

        - ``x`` -- Monomial in lyndon basis for free(e_i) or free(f_i). Must be
          an MONOMIAL, i.e. LieGenerator or LyndonBracket. Form by taking Lyn(-int-).list()[0][0]

        - ``k`` -- integer; what level we're acting on,

        - ``wt`` -- wt where the element will be acting

        - ``lvl_in_right`` -- when computing on a bracket [l,r](k), do we
          compute this as l(k)*r - r*l(k) if False, or l*r(k) - r(k)*l if True

        OUTPUT: matrix giving the action of this element on the weight space.

        EXAMPLES:

        This example illustrates the level one action of an imaginary root ::

            sage: M = matrix([[2, -2,0],[-2,2,-1], [0,-1,2]])
            sage: v = VectorSpace(QQ,3,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('a', 'b', 'c'))
            sage: base_wt = (V.vec((0,0,1)), -2)
            sage: root_space = V._root_system.root_space_prebasis((1,1,0))
            sage: x = root_space[0].list()[0][0];x
            [e0, e1]
            sage: V.frenkel_kac_matrix(x , 1, base_wt)
            [ 0  0  0]
            [ 0 -2  2]
            [ 1 -2  2]

        """
        e_or_f = x.to_word()[0][0]
        free = self._root_system._LyndonE if e_or_f == 'e' else self._root_system._LyndonF
        gens = [g.list()[0][0] for g in free.gens()]
        if x in gens:
            return self.frenkel_kac_simple_matrix(e_or_f, gens.index(x), k, wt)
        else:
            # compute [l, r](k) by actual matrix 'commutator'
            l = x._left
            r = x._right # dump whole level onto the right term. Why not?
            if lvl_in_right:
                return self._commutator_at_wt(l, 0, r, k, wt)
            else:
                return self._commutator_at_wt(l, k, r, 0, wt)

    def _test_serre_relations(self, e_or_f, k, wt, printing = True):
        #tests the serre relations at the gien weight and level of root
        rels = self._root_system.serre_relations_e if e_or_f == 'e' else self._root_system.serre_relations_f
        for rel in rels:
            o = self.frenkel_kac_matrix(rel, k, wt)
            if printing:
                print(f"for rel = {rel} we get matrix\n{o}")
            if o != 0:
                print(f'Relation{rel} failed to hold at wt {wt}!')
                return False
        return (True)

    def kac_moody_element_matrix(self,x,k, wt):
        r"""
        Given an element of the root space of the algebra, act by decomposing and acting by monomials

        INPUT:

        - ``x`` -- element in LyndonE or LyndonF of self._root_system

        - ``k`` -- lvl at which the element acts

        - ``wt`` -- what weight are we acting on?

        OUTPUT: the matrix of the action.

        EXAMPLES:

        We must pass this to LyndonE or LyndonF ::

            sage: M = matrix([[2,-1],[-1,2]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: H = HeisenbergAlgebraVSpace(QQ, v, labels = ('c', 'd'))
            sage: F = FockModule(QQ, v, labels=('c','d'))
            sage: V = LatticeVOA(QQ, v, labels = ('c', 'd'))
            sage: x = V._root_system._LyndonE( V._root_system.root_space_basis((1,0))[0]);x
            e0
            sage: V.kac_moody_element_matrix(7*x, 1, (V.vec((0,0)),-3))
            [ 21   7   0 -28   0  28   0 -14   0   0]
            [  7   7   7  -7  -7   0  28 -14  14 -14]


        """
        xs = x.list()
        if xs == [0]:
            return 0
        else:
            return sum([c * mx for (mx, c) in [(self.frenkel_kac_matrix(m, k, wt),c) for (m, c) in xs] if mx != 0])

    def composition_of_fourier_coeffs(self, xas, wt):
        r"""
        Given a list of Fourier coefficients, returns their succeccisve compositions without
        having to calculate intermediate weights by hand.

        INPUT:

        - ``xas`` -- list [(α_i, k_i)] of pairs, (vector, integer)

        - ``wt`` -- which weight we act at

        OUTPUT: The matrix of the operator composition of all of these.

        EXAMPLES:

        An example ::

            sage: M = matrix([[2,-1],[-1,2]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('c', 'd'))
            sage: a0 = V.vec((1,0))
            sage: a1 = V.vec((0,1))
            sage: xas = [(a0+ a1, 0), (-a0-a1, 0)]; xas
            [((1, 1), 0), ((-1, -1), 0)]
            sage: xas[::-1]
            [((-1, -1), 0), ((1, 1), 0)]
            sage: t1 = V.composition_of_fourier_coeffs(xas, (V.vec((1,0)),-3))
            sage: t2 = V.composition_of_fourier_coeffs(xas[::-1], (V.vec((1,0)),-3))
            sage: t2 - t1
            [1 0 0 0 0]
            [0 1 0 0 0]
            [0 0 1 0 0]
            [0 0 0 1 0]
            [0 0 0 0 1]

        """
        rev = xas[::-1]
        alphas = [self.vec(xa[0]) for xa in rev]
        ks = [xa[1] for xa in rev]
        cr = list(wt)
        for (xa, xk) in rev:
            cr[0] += self.vec(xa)
            cr[1] += xk
            npts = self._n_pts(cr)
            if npts <0 :
                return 0
        total_wt_difference = (sum(alphas), sum(ks))
        M = 1
        curr_wt = wt
        for (a, k) in rev:
            if M != 0:
                tgt = (self.vec(curr_wt[0] + self.vec(a)), curr_wt[1] + k)
                M = self.fourier_coefficient_matrix(self.vec(a), k, curr_wt)*M
                curr_wt = tgt
        return M

    def nested_commutator_of_fourier_coeffs(self, bracket, wt):
        r"""
        Returns the fourier coefficient on a bracket, where bracketed elements are lists of tuples of 'wts'

        INPUT:

        - ``bracket`` -- binary tree - list of pairs (α, k); the bracket [ [(α_i, k_i)], []] etc.

        - ``wt`` -- which weight to calculate at

        OUTPUT: matrix for the appropriate commutator.

        EXAMPLES:

        This simple Serre relation check in \hat sl2::

            sage: M = matrix([[2,-1],[-1,2]])
            sage: v = VectorSpace(QQ,2,inner_product_matrix=M)
            sage: V = LatticeVOA(QQ, v, labels = ('c', 'd'))
            sage: V.nested_commutator_of_fourier_coeffs( [((1,0),0),  [ ((1,0), -1), ((0,1), 0) ]]   , (self.vec((1,1)), -3))
            0

        """
        return( sum(m  for m in [ self.composition_of_fourier_coeffs(list(m), wt)* c
                          for (m, c) in LatticeVOA.lift_bracket(bracket).items()]
                        if m != 0))

    @staticmethod
    def lift_bracket(bkt):
        if not isinstance(bkt, list):
            return {(bkt,) : 1}
        else:
            l = bkt[0]
            r = bkt[1]
            lift_l = LatticeVOA.lift_bracket(l)
            lift_r = LatticeVOA.lift_bracket(r)
            out = {}
            for lmon, lc in lift_l.items():
                for rmon, rc in lift_r.items():
                    if lmon != rmon:
                        if lmon +  rmon in out:
                            out[lmon+rmon] += lc*rc
                        else:
                            out[lmon+rmon] = lc*rc
                        if rmon + lmon in out:
                            out[rmon+lmon] -= lc*rc
                        else:
                            out[rmon+lmon] = -lc*rc
            return out


# %% In[test sl3]

# %% In[]
## %% In[Test BKM]
#M = matrix([[0,-1],[-1,2]])
#v = VectorSpace(QQ,2,inner_product_matrix=M)
#H = HeisenbergAlgebraVSpace(QQ, v, labels = ('c', 'd'))
#F = FockModule(QQ, v, labels=('c','d'))
#V = LatticeVOA(QQ, v, labels = ('c', 'd'))
#
#base_wt = (vector((0,1)), -3)
#wts = [(vector((i,j)), -l) for i in range(2) for j in range(2) for k in range(2) for l in range(4) ]
#x = V.root_system.root_space_prebasis((1,1))[0].list()[0][0];x
#y = V.root_system.root_space_prebasis((1,0))[0].list()[0][0];y
#V._test_commutator_at_wts_if_bdd(x, 1, y, 2, wts, highest = 5)
#V._test_commutator_at_wts_if_bdd(x, 2, y, 1, wts, highest = 5)
#
## %% In[test on Lorenzian KM]
#M = matrix([[2, -2,0],[-2,2,-1], [0,-1,2]])
#v = VectorSpace(QQ,3,inner_product_matrix=M)
#H = HeisenbergAlgebraVSpace(QQ, v, labels = ('a', 'b', 'c'))
#F = FockModule(QQ, v, labels=('a','b', 'c'))
#V = LatticeVOA(QQ, v, labels = ('a', 'b', 'c'))
#wt = (vector((0,1,0)), -2)
#
#ms = V._root_space_casimir_summands_at_wt(vector((2,2,1)), 2, wt); ms
list(2*i for i in range(5))
M = Matrix([[2,-2,0],[-2,2,-1],[0,-1,2]])
M.det()
