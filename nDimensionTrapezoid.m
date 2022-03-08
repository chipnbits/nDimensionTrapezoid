%% A trapezoid approximation function for n-dimensions

% Specify the number of dimensions here
n = 4;

% number of partitions for each dimension, note that each dimension can have its
% own parameters assigned if desired.
Nk= 100*ones(1,n); %vector of number of points for each dimension 1 to k
nRanges.min = zeros(1,n); %min value vector for each dimension 1 to k
nRanges.max = ones(1,n); %max value for each dimension 1 to k

fun = @expForVararg; % e^(-1*x1*x2*x3*...xk) exponentiation function for n-dimensions

% create the variables and their spaced vectors:
nvars = cell(1,n);
for i = 1:n
    nvars{i} = linspace(nRanges.min(i),nRanges.max(i),Nk(i)+1);
end

% form a meshgrid for all dimensions and evaluate the function at those points
P = cell(1,n);
[P{:}] = ndgrid(nvars{:});

I = trapNdim(fun(P{:}), nvars{:})

%% Exponent Function for n-dimensions
% Function that takes a variable number of arguments to exponentiate
function x = expForVararg(varargin)
    prod = varargin{1};    
    for i = 2:nargin
        prod = prod.*varargin{i};
    end    
    x=exp(-1*prod);
end

%% Trapezoid integration for n-dimensions:
% A funtion that takes an n-dimension matrix with n vectors describing the grid
% spacing over the domain of trapezoidal approximation
%  trapNdim(F, x1, x2, ..., x_(n-1), xn)
% Inputs: F = m x o x p x q ...  n-dimension matrix of the function values
%         varargin = n number of vectors describing grid points for each dim 
%                   i.e. x1, x2, x3, x4, x5 are 5 vectors with grid points
function I = trapNdim(F, varargin)
    n = nargin-1;
    
    delnVars = cellfun(@(x) diff(x), varargin,'UniformOutput',false);
    % Make a cell array of all the indexes for the average computations
    % The nidxplus array contains the pair of 1:end-1 and 2:end index vectors
    % for each dimension of the F matrix.
    nidx = cellfun(@(x) 1:length(x), delnVars,'UniformOutput',false);
    nidx = cellfun(@(x) {x, x+1}, nidx,'UniformOutput',false);
    %nidx contains the upper and lower indices for each dimension
    
    % Use the allcomb function downloaded from  mathworks.com to create all the
    % permutations required of indexes for calculating the trapezoidal average
    % this creates 2^n matrices containing each "corner" of the n-dimension
    % trapezoid.  Divide by the number of corners (2^n) to get the average
    nidxPermutated = allcomb(nidx{:});
    numPerms = size(nidxPermutated,1);
    Fave = F(nidxPermutated{1,:});    
    for i = 2:numPerms
        Fave = Fave+ F(nidxPermutated{i,:});
    end
    Fave = Fave/numPerms;
    
    % Collapse the dimensions of the Fave matrix recursively from the 
    % highest dimension back to a scalar   
    % Multiply each layer of Fave along the ith dimension by the delta
    % values for the respective ith dimension vector and compact the layers
    % down.  The for loop reduces Fave by 1 dimension on
    % each iteration until it is a 1-D vector
    for i = n:-1:2
        %reshape the ith dimension delta vector along its proper dimension
         deltaith = reshape(delnVars{i},[ones(1,i-1) length(delnVars{i})]);
         % multiply into the array dimension and use summation along the 
         % same dimension to reduce dimension of Fave by one each step
         Fave = sum(Fave.*deltaith,i) ;            
    end
    % Final multiplication doesn't require reshape, just a dot product instead
    Fave = dot(Fave,delnVars{1});
    
    I=Fave;     
end


%% Allcomb from https://www.mathworks.com/matlabcentral/fileexchange/10064-allcomb-varargin
function A = allcomb(varargin)
% ALLCOMB - All combinations
%    B = ALLCOMB(A1,A2,A3,...,AN) returns all combinations of the elements
%    in the arrays A1, A2, ..., and AN. B is P-by-N matrix where P is the product
%    of the number of elements of the N inputs. 
%    This functionality is also known as the Cartesian Product. The
%    arguments can be numerical and/or characters, or they can be cell arrays.
%
%    Examples:
%       allcomb([1 3 5],[-3 8],[0 1]) % numerical input:
%       % -> [ 1  -3   0
%       %      1  -3   1
%       %      1   8   0
%       %        ...
%       %      5  -3   1
%       %      5   8   1 ] ; % a 12-by-3 array
%
%       allcomb('abc','XY') % character arrays
%       % -> [ aX ; aY ; bX ; bY ; cX ; cY] % a 6-by-2 character array
%
%       allcomb('xy',[65 66]) % a combination -> character output
%       % -> ['xA' ; 'xB' ; 'yA' ; 'yB'] % a 4-by-2 character array
%
%       allcomb({'hello','Bye'},{'Joe', 10:12},{99999 []}) % all cell arrays
%       % -> {  'hello'  'Joe'        [99999]
%       %       'hello'  'Joe'             []
%       %       'hello'  [1x3 double] [99999]
%       %       'hello'  [1x3 double]      []
%       %       'Bye'    'Joe'        [99999]
%       %       'Bye'    'Joe'             []
%       %       'Bye'    [1x3 double] [99999]
%       %       'Bye'    [1x3 double]      [] } ; % a 8-by-3 cell array
%
%    ALLCOMB(..., 'matlab') causes the first column to change fastest which
%    is consistent with matlab indexing. Example: 
%      allcomb(1:2,3:4,5:6,'matlab') 
%      % -> [ 1 3 5 ; 1 4 5 ; 1 3 6 ; ... ; 2 4 6 ]
%
%    If one of the N arguments is empty, ALLCOMB returns a 0-by-N empty array.
%    
%    See also NCHOOSEK, PERMS, NDGRID
%         and NCHOOSE, COMBN, KTHCOMBN (Matlab Central FEX)
% Tested in Matlab R2015a and up
% version 4.2 (apr 2018)
% (c) Jos van der Geest
% email: samelinoa@gmail.com
% History
% 1.1 (feb 2006), removed minor bug when entering empty cell arrays;
%     added option to let the first input run fastest (suggestion by JD)
% 1.2 (jan 2010), using ii as an index on the left-hand for the multiple
%     output by NDGRID. Thanks to Jan Simon, for showing this little trick
% 2.0 (dec 2010). Bruno Luong convinced me that an empty input should
% return an empty output.
% 2.1 (feb 2011). A cell as input argument caused the check on the last
%      argument (specifying the order) to crash.
% 2.2 (jan 2012). removed a superfluous line of code (ischar(..))
% 3.0 (may 2012) removed check for doubles so character arrays are accepted
% 4.0 (feb 2014) added support for cell arrays
% 4.1 (feb 2016) fixed error for cell array input with last argument being
%     'matlab'. Thanks to Richard for pointing this out.
% 4.2 (apr 2018) fixed some grammar mistakes in the help and comments
narginchk(1,Inf) ;
NC = nargin ;
% check if we should flip the order
if ischar(varargin{end}) && (strcmpi(varargin{end}, 'matlab') || strcmpi(varargin{end}, 'john'))
    % based on a suggestion by JD on the FEX
    NC = NC-1 ;
    ii = 1:NC ; % now first argument will change fastest
else
    % default: enter arguments backwards, so last one (AN) is changing fastest
    ii = NC:-1:1 ;
end
args = varargin(1:NC) ;
if any(cellfun('isempty', args)) % check for empty inputs
    warning('ALLCOMB:EmptyInput','One of more empty inputs result in an empty output.') ;
    A = zeros(0, NC) ;
elseif NC == 0 % no inputs
    A = zeros(0,0) ; 
elseif NC == 1 % a single input, nothing to combine
    A = args{1}(:) ; 
else
    isCellInput = cellfun(@iscell, args) ;
    if any(isCellInput)
        if ~all(isCellInput)
            error('ALLCOMB:InvalidCellInput', ...
                'For cell input, all arguments should be cell arrays.') ;
        end
        % for cell input, we use to indices to get all combinations
        ix = cellfun(@(c) 1:numel(c), args, 'un', 0) ;
        
        % flip using ii if last column is changing fastest
        [ix{ii}] = ndgrid(ix{ii}) ;
        
        A = cell(numel(ix{1}), NC) ; % pre-allocate the output
        for k = 1:NC
            % combine
            A(:,k) = reshape(args{k}(ix{k}), [], 1) ;
        end
    else
        % non-cell input, assuming all numerical values or strings
        % flip using ii if last column is changing fastest
        [A{ii}] = ndgrid(args{ii}) ;
        % concatenate
        A = reshape(cat(NC+1,A{:}), [], NC) ;
    end
end
end
